import re
import unicodedata
import zipfile
from io import BytesIO
from pathlib import Path

import requests
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from adjustText import adjust_text


# =========================================================
# 1. BEÁLLÍTÁSOK
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "DATA" / "Csehorszag"  / "Csehország_ábra-jarascsv.csv"
OUTPUT_PATH = BASE_DIR / "Media" / "Csehorszag" / "csehorszag_jarasok_vonzaskorzetek.png"

# GADM 4.1, 2. szint = járások (okres). A letöltött fájlt helyben
# elmentjük, így a további futásokhoz nem kell internet.
GEO_URL = (
    "https://geodata.ucdavis.edu/gadm/gadm4.1/json/"
    "gadm41_CZE_2.json.zip"
)
CACHE_PATH = BASE_DIR / "gadm41_CZE_2.json"

# Ha saját járáshatár-fájlod van, add meg ide az elérési útját.
LOCAL_GEO_PATH = None  # pl. BASE_DIR / "sajat_okresek.geojson"

# EPSG:5514 = S-JTSK / Krovak East North (cseh nemzeti vetület).
TARGET_EPSG = 5514

FIGSIZE = (16, 10)
DPI = 300

# Ismert névváltozatok (normalizált alakban) -> a CSV-beli normalizált név.
NAME_ALIASES = {
    "hlavnimestopraha": "praha",
    "prague": "praha",
    "praga": "praha",
}


# =========================================================
# 2. SEGÉDFÜGGVÉNYEK
# =========================================================

def normalize_name(value):
    """
    Járásnevek egységesítése az összekapcsoláshoz.

    - Eltávolítja az ékezeteket (Plzeň -> plzen).
    - Kisbetűsít.
    - Eltávolít minden nem betű/szám karaktert (szóköz, kötőjel),
      így a "Brno-město" és a "Brno-Mesto" ugyanaz a kulcs lesz.
    """
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^0-9a-z]", "", value.casefold())
    return NAME_ALIASES.get(value, value)


def load_districts():
    """Cseh járáshatárok betöltése (gyorsítótárral)."""
    if LOCAL_GEO_PATH is not None:
        return gpd.read_file(LOCAL_GEO_PATH)

    if CACHE_PATH.exists():
        return gpd.read_file(CACHE_PATH)

    print(f"Letöltés: {GEO_URL}")
    response = requests.get(
        GEO_URL,
        timeout=120,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    response.raise_for_status()

    with zipfile.ZipFile(BytesIO(response.content)) as zf:
        member = next(
            name for name in zf.namelist()
            if name.lower().endswith(".json")
        )
        CACHE_PATH.write_bytes(zf.read(member))

    print(f"Gyorsítótár elmentve: {CACHE_PATH}")

    return gpd.read_file(CACHE_PATH)


# =========================================================
# 3. CSV BEOLVASÁSA
# =========================================================

if not CSV_PATH.exists():
    raise FileNotFoundError(f"A CSV-fájl nem található: {CSV_PATH}")

df = pd.read_csv(
    CSV_PATH,
    encoding="utf-8-sig",
    sep=";"
)

df.columns = df.columns.str.strip()

df = df.rename(columns={
    "LAU1-kód": "lau1",
    "Okres kódja": "okres_kod",
    "Okres": "jaras",
    "Domináns központi okres": "kozpont",
})

required_columns = {"jaras", "kozpont"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        "A CSV oszlopai nem megfelelőek.\n"
        f"Elvárt oszlopok: {sorted(required_columns)}\n"
        f"Talált oszlopok: {df.columns.tolist()}"
    )

df = df.dropna(subset=["jaras", "kozpont"]).copy()

df["jaras"] = df["jaras"].astype(str).str.strip()
df["kozpont"] = df["kozpont"].astype(str).str.strip()

df["join_key"] = df["jaras"].map(normalize_name)

df = df[
    (df["jaras"] != "") &
    (df["kozpont"] != "") &
    (df["join_key"] != "")
].copy()

# Járásszám központonként.
counts = df.groupby("kozpont")["jaras"].nunique()

print("CSV oszlopnevei:", df.columns.tolist())
print("CSV-ben szereplő járások:", df["jaras"].nunique())
print("Vonzásközpontok száma:", df["kozpont"].nunique())


# =========================================================
# 4. DUPLIKÁLT KULCSOK ELLENŐRZÉSE
# =========================================================

center_counts = df.groupby("join_key")["kozpont"].nunique()
conflicting_keys = center_counts[center_counts > 1].index

if len(conflicting_keys) > 0:
    print("\nHIBA: Azonos névkulcshoz több vonzásközpont tartozik.")
    print(
        df[df["join_key"].isin(conflicting_keys)]
        [["jaras", "kozpont", "join_key"]]
        .sort_values(["join_key", "kozpont"])
        .to_string(index=False)
    )
    raise ValueError("A CSV-ben ütköző járáskulcsok vannak.")

df_unique = df.drop_duplicates(subset="join_key", keep="first").copy()

print("Egyedi járáskulcsok száma:", df_unique["join_key"].nunique())


# =========================================================
# 5. JÁRÁSHATÁROK BETÖLTÉSE
# =========================================================

gdf = load_districts()

print("\nGeoJSON oszlopai:")
print(gdf.columns.tolist())

name_candidates = ["NAME_2", "name", "NAME", "NAZEV", "okres", "Okres"]

name_col = next(
    (col for col in name_candidates if col in gdf.columns),
    None
)

if name_col is None:
    raise ValueError(
        "Nem található a járásnév oszlopa. "
        f"A GeoJSON oszlopai: {gdf.columns.tolist()}"
    )

gdf = gdf.dropna(subset=[name_col]).copy()

gdf["join_key"] = gdf[name_col].map(normalize_name)

geo_dupes = gdf["join_key"].duplicated(keep=False)

if geo_dupes.any():
    print("\nFIGYELEM: ismétlődő járásnevek a térképfájlban:")
    print(gdf.loc[geo_dupes, name_col].tolist())


# =========================================================
# 6. CSV ÉS TÉRKÉP ÖSSZEKAPCSOLÁSA
# =========================================================

map_df = gdf.merge(
    df_unique[["join_key", "jaras", "kozpont"]],
    on="join_key",
    how="left",
    validate="many_to_one"
)

missing_geo = (
    map_df.loc[map_df["kozpont"].isna(), name_col].drop_duplicates()
)

if not missing_geo.empty:
    print("\nTérképi járások CSV-rekord nélkül:")
    print(missing_geo.to_list())

missing_csv = sorted(set(df_unique["join_key"]) - set(gdf["join_key"]))

if missing_csv:
    print("\nCSV-s járások térképi elem nélkül (normalizált kulcsok):")
    print(missing_csv)

map_df = map_df.dropna(subset=["kozpont"]).copy()

if map_df.empty:
    raise ValueError(
        "Egy járás sem párosítható. "
        "Ellenőrizd a járásneveket a CSV-ben és a térképfájlban."
    )

print("\nTérképre került járások száma:", len(map_df))
print("Térképen szereplő vonzásközpontok:", map_df["kozpont"].nunique())


# =========================================================
# 7. KOORDINÁTARENDSZER
# =========================================================

if map_df.crs is None:
    raise ValueError("A térképfájl koordinátarendszere nincs megadva.")

map_df = map_df.to_crs(epsg=TARGET_EPSG)


# =========================================================
# 8. SZÍNEK ÉS TÉRKÉP
# =========================================================

centers = sorted(map_df["kozpont"].unique())

palette = plt.get_cmap("tab20").colors

color_map = {
    center: palette[i % len(palette)]
    for i, center in enumerate(centers)
}

fig, ax = plt.subplots(figsize=FIGSIZE)

for center in centers:
    subset = map_df[map_df["kozpont"] == center]

    subset.plot(
        ax=ax,
        color=color_map[center],
        edgecolor="white",
        linewidth=0.6
    )

# A vonzáskörzetek külső határa vastagabb vonallal.
catchments = map_df.dissolve(by="kozpont")
catchments.boundary.plot(ax=ax, color="black", linewidth=0.9, zorder=4)


# =========================================================
# 9. VONZÁSKÖZPONTOK ÉS FELIRATOK
# =========================================================

texts = []

for center in centers:
    subset = map_df[map_df["kozpont"] == center]

    # A központ saját járását keressük a körzetén belül.
    center_key = normalize_name(center)

    center_rows = subset[subset["join_key"] == center_key]

    if center_rows.empty:
        point = subset.geometry.union_all().representative_point()
    else:
        point = center_rows.geometry.iloc[0].representative_point()

    ax.scatter(point.x, point.y, s=20, color="black", zorder=5)

    label = f"{center} ({counts.get(center, 0)})"

    texts.append(
        ax.text(
            point.x,
            point.y,
            label,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
            color="black",
            zorder=6,
            bbox=dict(
                facecolor="white",
                alpha=0.75,
                edgecolor="none",
                pad=1.2
            )
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
    "Csehország járásainak vonzáskörzetei\n"
    "a Huff-modell alapján",
    fontsize=16,
    fontweight="bold",
    loc="left",
    pad=15
)

ax.axis("off")
ax.set_aspect("equal")

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=DPI,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print(f"\nA térkép elkészült: {OUTPUT_PATH}")