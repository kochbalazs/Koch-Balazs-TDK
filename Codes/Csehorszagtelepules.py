import re
import unicodedata
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

CSV_PATH = BASE_DIR / "DATA" / "Csehorszag" / "Csehország_ábra-településcsv.csv"
OUTPUT_PATH = BASE_DIR / "Media" / "Csehorszag" / "csehorszag_telepulesek_vonzaskorzetek.png"

# Eurostat GISCO LAU (települési szintű) határok. A fájl az egész EU-t
# tartalmazza (nagy, több száz MB), ezért az első futáskor letöltjük,
# kiszűrjük belőle Csehországot, és a kis méretű változatot elmentjük.
# A további futások már csak ezt a gyorsítótárat használják.
GEO_URL = (
    "https://gisco-services.ec.europa.eu/distribution/v2/lau/geojson/"
    "LAU_RG_01M_2021_4326.geojson"
)
RAW_GEO_PATH = BASE_DIR / "LAU_RG_01M_2021_4326.geojson"
CACHE_PATH = BASE_DIR / "cz_lau_telepulesek.geojson"

# Ha már van saját, csehországi településhatár-fájlod, add meg ide
# az elérési útját, és a letöltés kimarad.
LOCAL_GEO_PATH = None  # pl. BASE_DIR / "sajat_telepulesek.geojson"

# EPSG:5514 = S-JTSK / Krovak East North (cseh nemzeti vetület).
TARGET_EPSG = 5514

FIGSIZE = (16, 10)
DPI = 300


# =========================================================
# 2. SEGÉDFÜGGVÉNYEK
# =========================================================

def normalize_name(value):
    """Településnév egységesítése (kisbetű, felesleges szóközök nélkül)."""
    value = str(value).strip().casefold()
    value = re.sub(r"\s+", " ", value)
    return value


def normalize_code(value):
    """Településkód egységesítése: szöveg, szóközök és '.0' nélkül."""
    value = str(value).strip()
    value = re.sub(r"\.0$", "", value)
    return value


def download_file(url, target_path):
    """Nagy fájl letöltése darabokban, hogy ne töltsük a memóriába."""
    print(f"Letöltés: {url}")
    with requests.get(
        url,
        stream=True,
        timeout=120,
        headers={"User-Agent": "Mozilla/5.0"}
    ) as response:
        response.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)


def load_czech_municipalities():
    """Csehországi településhatárok betöltése (gyorsítótárral)."""
    if LOCAL_GEO_PATH is not None:
        return gpd.read_file(LOCAL_GEO_PATH)

    if CACHE_PATH.exists():
        return gpd.read_file(CACHE_PATH)

    if not RAW_GEO_PATH.exists():
        download_file(GEO_URL, RAW_GEO_PATH)

    raw = gpd.read_file(RAW_GEO_PATH)

    if "CNTR_CODE" not in raw.columns:
        raise ValueError(
            "A letöltött fájlban nincs CNTR_CODE oszlop. "
            f"Oszlopok: {raw.columns.tolist()}"
        )

    cz = raw[raw["CNTR_CODE"] == "CZ"].copy()

    if cz.empty:
        raise ValueError("A GISCO-fájlban nem található cseh település.")

    cz.to_file(CACHE_PATH, driver="GeoJSON")
    print(f"Csehországi gyorsítótár elmentve: {CACHE_PATH}")

    return cz


# =========================================================
# 3. CSV BEOLVASÁSA
# =========================================================

if not CSV_PATH.exists():
    raise FileNotFoundError(f"A CSV-fájl nem található: {CSV_PATH}")

df = pd.read_csv(
    CSV_PATH,
    encoding="utf-8-sig",
    sep=";",
    dtype={"KOD": str}
)

df.columns = df.columns.str.strip()

df = df.rename(columns={
    "KOD": "kod",
    "Település": "telepules",
    "Domináns központ (Huff)": "kozpont",
})

required_columns = {"kod", "telepules", "kozpont"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        "A CSV oszlopai nem megfelelőek.\n"
        f"Elvárt oszlopok: {sorted(required_columns)}\n"
        f"Talált oszlopok: {df.columns.tolist()}"
    )

df = df.dropna(subset=["kod", "telepules", "kozpont"]).copy()

df["kod"] = df["kod"].map(normalize_code)
df["telepules"] = df["telepules"].astype(str).str.strip()
df["kozpont"] = df["kozpont"].astype(str).str.strip()

df = df[
    (df["kod"] != "") &
    (df["telepules"] != "") &
    (df["kozpont"] != "")
].copy()

# Településszám központonként.
counts = df.groupby("kozpont")["kod"].nunique()

print("CSV oszlopnevei:", df.columns.tolist())
print("CSV-ben szereplő települések (kód szerint):", df["kod"].nunique())
print("Vonzásközpontok száma:", df["kozpont"].nunique())


# =========================================================
# 4. DUPLIKÁLT KULCSOK ELLENŐRZÉSE
# =========================================================

# A cseh településnevek gyakran ismétlődnek (pl. több "Lhota"),
# ezért NEM név, hanem településkód alapján kapcsolunk össze.

center_counts = df.groupby("kod")["kozpont"].nunique()
conflicting_codes = center_counts[center_counts > 1].index

if len(conflicting_codes) > 0:
    print("\nHIBA: Azonos kódhoz több vonzásközpont tartozik.")
    print(
        df[df["kod"].isin(conflicting_codes)]
        .sort_values(["kod", "kozpont"])
        .to_string(index=False)
    )
    raise ValueError("A CSV-ben ütköző településkódok vannak.")

df_unique = df.drop_duplicates(subset="kod", keep="first").copy()

print("Egyedi településkódok száma:", df_unique["kod"].nunique())


# =========================================================
# 5. TELEPÜLÉSHATÁROK BETÖLTÉSE
# =========================================================

gdf = load_czech_municipalities()

print("\nGeoJSON oszlopai:")
print(gdf.columns.tolist())

# A településkód mezőjének meghatározása.
code_candidates = ["LAU_ID", "LAU_CODE", "KOD", "kod", "KOD_OBEC", "code"]

code_col = next(
    (col for col in code_candidates if col in gdf.columns),
    None
)

if code_col is not None:
    gdf["join_code"] = gdf[code_col].map(normalize_code)
elif "GISCO_ID" in gdf.columns:
    # A GISCO_ID formátuma: "CZ_500011" -> a kód az aláhúzás után van.
    gdf["join_code"] = (
        gdf["GISCO_ID"].astype(str).str.split("_").str[-1].map(normalize_code)
    )
    code_col = "GISCO_ID"
else:
    raise ValueError(
        "Nem található a településkód oszlopa. "
        f"A GeoJSON oszlopai: {gdf.columns.tolist()}"
    )

print("Használt kódoszlop:", code_col)

# A településnév csak a diagnosztikához kell.
name_candidates = ["LAU_NAME", "NAME", "name", "NAZEV", "nev"]
name_col = next(
    (col for col in name_candidates if col in gdf.columns),
    None
)

if name_col is None:
    gdf["_nev"] = gdf["join_code"]
    name_col = "_nev"


# =========================================================
# 6. CSV ÉS TÉRKÉP ÖSSZEKAPCSOLÁSA
# =========================================================

map_df = gdf.merge(
    df_unique[["kod", "telepules", "kozpont"]],
    left_on="join_code",
    right_on="kod",
    how="left",
    validate="many_to_one"
)

missing_geo = (
    map_df.loc[map_df["kozpont"].isna(), [name_col, "join_code"]]
    .drop_duplicates()
)

if not missing_geo.empty:
    print("\nTérképi elemek CSV-rekord nélkül (példák):")
    print(missing_geo.head(20).to_string(index=False))
    print("Számuk:", len(missing_geo))

missing_csv = set(df_unique["kod"]) - set(gdf["join_code"])

if missing_csv:
    print("\nCSV-s kódok térképi elem nélkül (példák):")
    print(sorted(missing_csv)[:20])
    print("Számuk:", len(missing_csv))

map_df = map_df.dropna(subset=["kozpont"]).copy()

if map_df.empty:
    raise ValueError(
        "Egy település sem párosítható. "
        "Ellenőrizd a kódokat a CSV-ben és a térképfájlban."
    )

print("\nTérképre került geometriai elemek száma:", len(map_df))
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

# Kategorikus palettából veszünk színeket (nem nyújtjuk szét a skálát),
# így a szomszédos körzetek jól megkülönböztethetők maradnak.
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
        linewidth=0.12
    )

# A vonzáskörzetek külső határa vastagabb vonallal.
catchments = map_df.dissolve(by="kozpont")
catchments.boundary.plot(ax=ax, color="black", linewidth=0.6, zorder=4)


# =========================================================
# 9. VONZÁSKÖZPONTOK ÉS FELIRATOK
# =========================================================

texts = []

for center in centers:
    subset = map_df[map_df["kozpont"] == center]

    # A vonzásközpont saját településének keresése név alapján,
    # a saját körzetén belül. Több találatnál a legnagyobb területű.
    center_rows = subset[
        subset["telepules"].map(normalize_name) == normalize_name(center)
    ]

    if center_rows.empty:
        # Tartalék: a körzet egészének poligonon belüli pontja.
        point = subset.geometry.union_all().representative_point()
    else:
        biggest = center_rows.geometry.area.idxmax()
        point = center_rows.loc[biggest, "geometry"].representative_point()

    ax.scatter(point.x, point.y, s=14, color="black", zorder=5)

    label = f"{center} ({counts.get(center, 0)})"

    texts.append(
        ax.text(
            point.x,
            point.y,
            label,
            fontsize=8,
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
    "Csehország településeinek vonzáskörzetei\n"
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