import re
from pathlib import Path
from io import BytesIO

import requests
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from adjustText import adjust_text


# =========================================================
# 1. BEÁLLÍTÁSOK
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "DATA" / "Magyarorszag" / "orszagosbeosztas" / "Magyarországabrak2.csv"
OUTPUT_PATH = BASE_DIR / "Media" / "Magyarorszag" / "magyarorszag_vonzaskorzetekabrak2.png"

GEOJSON_URL = (
    "https://raw.githubusercontent.com/"
    "wuerdo/geoHungary/master/settlements.geojson"
)

FIGSIZE = (16, 10)
DPI = 300


# =========================================================
# 2. SEGÉDFÜGGVÉNYEK
# =========================================================

def normalize_name(value):
    """Kisbetűsítés, szóközök egységesítése, ékezetek megtartása."""
    value = str(value).strip().casefold()
    return re.sub(r"\s+", " ", value)


def normalize_ksh(value):
    """KSH-kód 5 számjegyűre hozva (vezető nullákkal)."""
    digits = re.sub(r"\D", "", str(value))
    return digits.zfill(5) if digits else ""


# =========================================================
# 3. CSV BEOLVASÁSA
# =========================================================

if not CSV_PATH.exists():
    raise FileNotFoundError(f"A CSV-fájl nem található: {CSV_PATH}")

df = pd.read_csv(
    CSV_PATH,
    sep=";",
    encoding="utf-8-sig",   # BOM-os UTF-8
    dtype=str
)

df.columns = df.columns.str.strip()

df = df.rename(columns={
    "Település": "telepules",
    "KSH kód": "ksh",
    "Domináns központ (Huff)": "kozpont",
})

required_columns = {"telepules", "ksh", "kozpont"}
if not required_columns.issubset(df.columns):
    raise ValueError(
        "A CSV oszlopai nem megfelelőek.\n"
        f"Elvárt oszlopok: {sorted(required_columns)}\n"
        f"Talált oszlopok: {df.columns.tolist()}"
    )

df = df.dropna(subset=["telepules", "kozpont"]).copy()

for col in ["telepules", "kozpont"]:
    df[col] = df[col].astype(str).str.strip()

df["ksh_key"] = df["ksh"].map(normalize_ksh)
df["name_key"] = df["telepules"].map(normalize_name)

df = df[(df["telepules"] != "") & (df["kozpont"] != "")].copy()

counts = df.groupby("kozpont")["telepules"].nunique()

print("CSV-ben szereplő települések:", df["telepules"].nunique())
print("Vonzásközpontok száma:", df["kozpont"].nunique())


# =========================================================
# 4. DUPLIKÁLT KULCSOK ELLENŐRZÉSE
# =========================================================

for key in ["ksh_key", "name_key"]:
    conflicts = df.groupby(key)["kozpont"].nunique()
    conflicts = conflicts[conflicts > 1].index
    if len(conflicts) > 0:
        print(f"\nHIBA: {key} több vonzásközponthoz tartozik:")
        print(
            df[df[key].isin(conflicts)][["telepules", "ksh", "kozpont"]]
            .sort_values("telepules")
            .to_string(index=False)
        )
        raise ValueError("Ütköző kulcsok a CSV-ben, javítsd az adatokat.")

df_ksh = df.drop_duplicates(subset="ksh_key").copy()
df_name = df.drop_duplicates(subset="name_key").copy()


# =========================================================
# 5. TELEPÜLÉSHATÁROK LETÖLTÉSE
# =========================================================

response = requests.get(GEOJSON_URL, timeout=60)
response.raise_for_status()

gdf = gpd.read_file(BytesIO(response.content))

print("\nGeoJSON oszlopai:", gdf.columns.tolist())

if gdf.crs is None:
    raise ValueError("A GeoJSON koordinátarendszere nincs megadva.")

name_candidates = [
    "NAME", "name", "NAME_HU", "name_hu", "telepules",
    "Település", "nev", "Név", "settlement", "municipality",
]
name_col = next((c for c in name_candidates if c in gdf.columns), None)

if name_col is None:
    raise ValueError(
        f"Nem található a településnév oszlopa. Oszlopok: {gdf.columns.tolist()}"
    )

gdf = gdf.dropna(subset=[name_col]).copy()
gdf["name_key"] = gdf[name_col].map(normalize_name)

# Budapest kerületei (ha külön szerepelnek) egy körzetbe kerüljenek.
gdf.loc[
    gdf["name_key"].str.startswith("budapest"), "name_key"
] = "budapest"

# KSH-kód jellegű oszlop keresése a GeoJSON-ban.
ksh_col = next(
    (c for c in gdf.columns
     if "ksh" in c.lower() or c.lower() in {"kodszam", "code", "kod", "kód"}),
    None
)


# =========================================================
# 6. CSV ÉS TÉRKÉP ÖSSZEKAPCSOLÁSA
# =========================================================

def merge_by_name():
    return gdf.merge(
        df_name[["name_key", "telepules", "kozpont"]],
        on="name_key", how="left"
    )

def merge_by_ksh():
    tmp = gdf.copy()
    tmp["ksh_key"] = tmp[ksh_col].map(normalize_ksh)
    return tmp.merge(
        df_ksh[["ksh_key", "telepules", "kozpont"]],
        on="ksh_key", how="left"
    )

map_name = merge_by_name()
best, method = map_name, "településnév"

if ksh_col is not None:
    map_ksh = merge_by_ksh()
    if map_ksh["kozpont"].notna().sum() > map_name["kozpont"].notna().sum():
        best, method = map_ksh, f"KSH-kód ({ksh_col})"

print(f"\nÖsszekapcsolás módja: {method}")

map_df = best

missing = map_df.loc[map_df["kozpont"].isna(), name_col].drop_duplicates()
if not missing.empty:
    print("Nem párosított GeoJSON-elemek száma:", len(missing))
    print("Példák:", missing.head(30).to_list())

matched_names = set(map_df.loc[map_df["kozpont"].notna(), "telepules"])
csv_missing = sorted(set(df["telepules"]) - matched_names)
print("CSV-települések, amelyek nem kerültek térképre:", len(csv_missing))
if csv_missing:
    print("Példák:", csv_missing[:30])

map_df = map_df.dropna(subset=["kozpont"]).copy()

if map_df.empty:
    raise ValueError("Egy település sem párosítható.")

print("Térképre került geometriai elemek:", len(map_df))
print("Térképen szereplő vonzásközpontok:", map_df["kozpont"].nunique())


# =========================================================
# 7. KOORDINÁTARENDSZER (EOV)
# =========================================================

map_df = map_df.to_crs(epsg=23700)


# =========================================================
# 8. SZÍNEK ÉS TÉRKÉP
# =========================================================

centers = sorted(map_df["kozpont"].unique())

cmap = plt.get_cmap("tab20", len(centers))
color_map = {c: cmap(i) for i, c in enumerate(centers)}

fig, ax = plt.subplots(figsize=FIGSIZE)

for center in centers:
    map_df[map_df["kozpont"] == center].plot(
        ax=ax,
        color=color_map[center],
        edgecolor="white",
        linewidth=0.12
    )


# =========================================================
# 9. VONZÁSKÖZPONTOK ÉS FELIRATOK
# =========================================================

texts = []

for center in centers:
    subset = map_df[map_df["kozpont"] == center]

    center_rows = subset[subset["name_key"] == normalize_name(center)]

    if not center_rows.empty:
        # A központ saját települése (Budapest kerületei egyben).
        geom = center_rows.geometry.union_all()
    else:
        # Nem település (pl. Balaton): a teljes körzet összevont alakzata.
        geom = subset.geometry.union_all()

    point = geom.representative_point()

    ax.scatter(point.x, point.y, s=10, color="black", zorder=5)

    label = f"{center} ({counts.get(center, 0)})"

    texts.append(
        ax.text(
            point.x, point.y, label,
            fontsize=7, fontweight="bold",
            ha="left", va="bottom", color="black", zorder=6,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.2)
        )
    )

adjust_text(
    texts,
    ax=ax,
    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5)
)


# =========================================================
# 10. FORMÁZÁS ÉS MENTÉS
# =========================================================

ax.set_title(
    "Magyarország településeinek vonzáskörzetei\na Huff-modell alapján",
    fontsize=16, fontweight="bold", loc="left", pad=15
)

ax.axis("off")
ax.set_aspect("equal")

plt.tight_layout()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.show()

print(f"\nA térkép elkészült: {OUTPUT_PATH}")