import re
import unicodedata
from pathlib import Path

import numpy as np
import requests
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from adjustText import adjust_text


# =========================================================
# 1. BEÁLLÍTÁSOK
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

BEOSZTAS_CSV = BASE_DIR /"DATA" / "Szlovakia" / "szlovakiabeosztascsv.csv"
URESEDES_CSV = BASE_DIR / "DATA" / "Szlovakia" / "Szlovákia_gyurukcsv.csv"
OUTPUT_PATH = BASE_DIR / "Media" / "Szlovakia" / "szlovakia_gyuruk_uresedes.png"

# Eurostat GISCO LAU (települési szintű) határok. A fájl az egész EU-t
# tartalmazza (nagy), ezért az első futáskor letöltjük, kiszűrjük belőle
# Szlovákiát, és a kis méretű változatot elmentjük.
GISCO_URL = (
    "https://gisco-services.ec.europa.eu/distribution/v2/lau/geojson/"
    "LAU_RG_01M_2021_4326.geojson"
)
RAW_GEO_PATH = BASE_DIR / "LAU_RG_01M_2021_4326.geojson"
CACHE_PATH = BASE_DIR / "sk_lau_telepulesek.geojson"

# Ha saját szlovákiai településhatár-fájlod van, add meg ide az elérési útját.
LOCAL_GEO_PATH = None  # pl. BASE_DIR / "sajat_telepulesek.geojson"

# EPSG:5514 = S-JTSK / Krovak East North (Szlovákiában is hivatalos).
TARGET_EPSG = 5514

# Sávhatárok (%). Az intervallumok balról zártak:
# pl. a pontosan 10,0 a "10–12" sávba kerül.
# (Ugyanazok, mint a lengyel és a magyar térképen, hogy összehasonlíthatók
# legyenek.)
BINS = [8, 10, 12, 15, 18, 21, 25]

# Világostól a sötétig (ColorBrewer YlOrRd, 8 osztály):
# minél nagyobb a nem lakott lakások aránya, annál erősebb a szín.
COLORS = [
    "#ffffcc", "#ffeda0", "#fed976", "#feb24c",
    "#fd8d3c", "#fc4e2a", "#e31a1c", "#b10026",
]

NO_DATA_COLOR = "#d9d9d9"

# True: a jelmagyarázatban minden sáv szerepel, akkor is, ha egyik
# település sem esik bele. False: csak a ténylegesen használt sávok.
SHOW_EMPTY_CLASSES = True

FIGSIZE = (16, 8)
DPI = 300


# =========================================================
# 2. SEGÉDFÜGGVÉNYEK
# =========================================================

# Pozsony és Kassa a térképfájlban városrészekre bontva szerepel
# (pl. "Bratislava - Staré Mesto"), a beosztásban viszont egy-egy sor.
CITY_PART_RE = re.compile(r"^\s*(Bratislava|Košice)(?![^\W\d_])")


def normalize_text(value):
    """Névkulcs: ékezetek, kis/nagybetű és minden nem betű/szám nélkül."""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[^0-9a-z]", "", value.casefold())


def normalize_code(value):
    """Településkód egységesítése: szöveg, szóközök és '.0' nélkül."""
    return re.sub(r"\.0$", "", str(value).strip())


def city_key(name):
    """Ha a név Pozsony vagy Kassa (városrésze), a város kulcsa, különben None."""
    name = unicodedata.normalize("NFC", str(name))
    match = CITY_PART_RE.match(name)
    return normalize_text(match.group(1)) if match else None


def ring_number(column_name):
    """'Központ' -> 0, 'Q3' -> 3."""
    name = str(column_name).strip()

    if name.casefold() == "központ":
        return 0

    match = re.match(r"[Qq](\d+)$", name)

    if match is None:
        raise ValueError(f"Ismeretlen gyűrű: {column_name}")

    return int(match.group(1))


def is_ring_column(column_name):
    name = str(column_name).strip()
    return name.casefold() == "központ" or re.match(r"[Qq]\d+$", name)


def bin_label(index):
    """A sávok felirata a jelmagyarázathoz."""
    def fmt(x):
        return f"{x:g}".replace(".", ",")

    if index == 0:
        return f"< {fmt(BINS[0])} %"

    if index == len(BINS):
        return f"≥ {fmt(BINS[-1])} %"

    return f"{fmt(BINS[index - 1])}–{fmt(BINS[index])} %"


def read_vacancy_table(path):
    """
    A gyűrűtábla beolvasása hosszú alakba (kozpont, gyuru, uresedes).
    A címsor előtti sorokat átugorjuk, a tizedesvesszőt kezeljük.
    """
    with open(path, encoding="utf-8-sig") as f:
        lines = f.read().splitlines()

    header_idx = next(
        (i for i, line in enumerate(lines) if "Vonzáskörzet" in line),
        None
    )

    if header_idx is None:
        raise ValueError(f"A 'Vonzáskörzet' fejléc nem található: {path}")

    header = [c.strip() for c in lines[header_idx].split(";")]

    rows = []

    for line in lines[header_idx + 1:]:
        cells = [c.strip() for c in line.split(";")]

        if cells[0] == "":
            break

        rows.append(cells)

    wide = pd.DataFrame(rows, columns=header)

    center_col = header[0]

    # Az összesítő sor nem körzet.
    wide = wide[
        ~wide[center_col].str.casefold().str.startswith("összes")
    ].copy()

    ring_cols = [c for c in header[1:] if is_ring_column(c)]

    long = wide.melt(
        id_vars=[center_col],
        value_vars=ring_cols,
        var_name="oszlop",
        value_name="uresedes"
    )

    long = long.rename(columns={center_col: "kozpont"})
    long["kozpont"] = long["kozpont"].str.strip()
    long["gyuru"] = long["oszlop"].map(ring_number)
    long["uresedes"] = pd.to_numeric(
        long["uresedes"].str.replace(",", ".", regex=False),
        errors="coerce"
    )

    # A táblában törtek vannak (0,15 = 15 %), a sávhatárok viszont
    # százalékban vannak megadva.
    if long["uresedes"].max() <= 1.0:
        print("Az üresedési értékek törtek (0–1), átváltás százalékra.")
        long["uresedes"] = long["uresedes"] * 100

    return long[["kozpont", "gyuru", "uresedes"]]


def load_settlements():
    """Szlovákiai településhatárok betöltése (gyorsítótárral)."""
    if LOCAL_GEO_PATH is not None:
        return gpd.read_file(LOCAL_GEO_PATH)

    if CACHE_PATH.exists():
        return gpd.read_file(CACHE_PATH)

    if not RAW_GEO_PATH.exists():
        print(f"Letöltés: {GISCO_URL}")
        with requests.get(
            GISCO_URL,
            stream=True,
            timeout=120,
            headers={"User-Agent": "Mozilla/5.0"}
        ) as response:
            response.raise_for_status()
            with open(RAW_GEO_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

    raw = gpd.read_file(RAW_GEO_PATH)

    if "CNTR_CODE" not in raw.columns:
        raise ValueError(
            "A letöltött fájlban nincs CNTR_CODE oszlop. "
            f"Oszlopok: {raw.columns.tolist()}"
        )

    sk = raw[raw["CNTR_CODE"] == "SK"].copy()

    if sk.empty:
        raise ValueError("A GISCO-fájlban nem található szlovák település.")

    sk.to_file(CACHE_PATH, driver="GeoJSON")
    print(f"Szlovákiai gyorsítótár elmentve: {CACHE_PATH}")

    return sk


# =========================================================
# 3. TÁBLÁZATOK BEOLVASÁSA
# =========================================================

for path in (BEOSZTAS_CSV, URESEDES_CSV):
    if not path.exists():
        raise FileNotFoundError(f"A CSV-fájl nem található: {path}")

settlements = pd.read_csv(
    BEOSZTAS_CSV,
    encoding="utf-8-sig",
    sep=";",
    dtype=str
)

settlements.columns = settlements.columns.str.strip()

settlements = settlements.rename(columns={
    "Település kódja": "kod",
    "Település": "telepules",
    "Domináns központ (Huff)": "kozpont",
    "Központ?": "is_kozpont",
    "Gyűrű": "gyuru_nev",
})

required = {"kod", "telepules", "kozpont", "gyuru_nev"}

if not required.issubset(settlements.columns):
    raise ValueError(
        "A beosztás-tábla oszlopai nem megfelelőek.\n"
        f"Elvárt: {sorted(required)}\n"
        f"Talált: {settlements.columns.tolist()}"
    )

settlements = settlements.dropna(subset=["kod", "kozpont", "gyuru_nev"]).copy()

settlements["kod"] = settlements["kod"].map(normalize_code)
settlements["telepules"] = settlements["telepules"].astype(str).str.strip()
settlements["kozpont"] = settlements["kozpont"].astype(str).str.strip()
settlements["gyuru"] = settlements["gyuru_nev"].map(ring_number)

vacancy = read_vacancy_table(URESEDES_CSV)

print("Települések száma a CSV-ben:", settlements["kod"].nunique())
print("Vonzáskörzetek száma:", settlements["kozpont"].nunique())
print("Üresedési értékek száma:", vacancy["uresedes"].notna().sum())


# =========================================================
# 4. ELLENŐRZÉSEK
# =========================================================

# A településnevek Szlovákiában gyakran ismétlődnek (pl. több "Belá"),
# ezért a térképpel NEM név, hanem településkód alapján kapcsolunk össze.

if settlements["kod"].duplicated().any():
    dupes = settlements.loc[settlements["kod"].duplicated(keep=False), "kod"]
    raise ValueError(
        "Ismétlődő településkódok a beosztás-táblában: "
        f"{sorted(dupes.unique())[:20]}"
    )

only_in_settlements = set(settlements["kozpont"]) - set(vacancy["kozpont"])
only_in_vacancy = set(vacancy["kozpont"]) - set(settlements["kozpont"])

if only_in_settlements or only_in_vacancy:
    print("\nFIGYELEM: a körzetnevek eltérnek a két táblában.")
    print("Csak a beosztás-táblában:", sorted(only_in_settlements))
    print("Csak az üresedési táblában:", sorted(only_in_vacancy))

# Minden település a saját körzetének és gyűrűjének értékét kapja.
data = settlements.merge(
    vacancy,
    on=["kozpont", "gyuru"],
    how="left",
    validate="many_to_one"
)

n_missing = data["uresedes"].isna().sum()

if n_missing > 0:
    print(f"\nFIGYELEM: {n_missing} településhez nincs üresedési érték:")
    print(
        data.loc[data["uresedes"].isna(), ["kozpont", "gyuru"]]
        .drop_duplicates()
        .to_string(index=False)
    )

# A városok (Pozsony, Kassa) kódja nem szám, hanem szöveg
# ("Bratislava (városrészekből)"), mert a térképen több városrészből állnak.
special = settlements[~settlements["kod"].str.fullmatch(r"\d+")]

special_codes = {
    normalize_text(row.telepules): row.kod
    for row in special.itertuples()
}

print("Városrészekből összeállított városok:", sorted(special_codes))


# =========================================================
# 5. TELEPÜLÉSHATÁROK BETÖLTÉSE ÉS ÖSSZEKAPCSOLÁS
# =========================================================

gdf = load_settlements()

print("\nGeoJSON oszlopai:")
print(gdf.columns.tolist())

code_candidates = ["LAU_ID", "LAU_CODE", "KOD", "kod", "id", "code"]
name_candidates = ["LAU_NAME", "NAME", "name", "NAZOV", "nazov", "NM"]

code_col = next((c for c in code_candidates if c in gdf.columns), None)
name_col = next((c for c in name_candidates if c in gdf.columns), None)

if code_col is not None:
    gdf["join_code"] = gdf[code_col].map(normalize_code)
elif "GISCO_ID" in gdf.columns:
    # A GISCO_ID formátuma: "SK_508454".
    gdf["join_code"] = gdf["GISCO_ID"].astype(str).str.split("_").str[-1].map(normalize_code)
    code_col = "GISCO_ID"
else:
    raise ValueError(
        "Nem található a településkód oszlopa. "
        f"A GeoJSON oszlopai: {gdf.columns.tolist()}"
    )

print("Használt kódoszlop:", code_col)

# 5/a. Pozsony és Kassa városrészeinek egyesítése egy-egy poligonná.
if name_col is not None and special_codes:
    csv_codes = set(settlements["kod"])

    keys = gdf[name_col].map(city_key)

    is_part = (
        keys.notna()
        & keys.isin(list(special_codes))
        & ~gdf["join_code"].isin(csv_codes)
    )

    if is_part.any():
        gdf.loc[is_part, "join_code"] = keys[is_part].map(special_codes)

        parts = gdf.loc[is_part, "join_code"].value_counts()

        for code, n in parts.items():
            print(f"  {code}: {n} városrész egyesítve")

        merged = gdf[is_part].dissolve(by="join_code", as_index=False)

        gdf = gpd.GeoDataFrame(
            pd.concat([gdf[~is_part], merged], ignore_index=True),
            geometry="geometry",
            crs=gdf.crs
        )
    else:
        print("\nFIGYELEM: nem találhatók Pozsony/Kassa városrészei "
              f"a(z) '{name_col}' oszlopban.")
elif special_codes:
    print("\nFIGYELEM: nincs névoszlop, Pozsony és Kassa városrészeit "
          "nem tudom egyesíteni.")

# 5/b. Összekapcsolás településkód alapján.
code_cov = len(set(settlements["kod"]) & set(gdf["join_code"])) / len(
    set(settlements["kod"])
)

print(f"\nKód alapú lefedettség: {code_cov:.1%}")

if code_cov == 0:
    raise ValueError(
        "Egy település sem párosítható kód alapján. "
        f"A GeoJSON oszlopai: {gdf.columns.tolist()}"
    )

map_df = gdf.merge(
    data[["kod", "telepules", "kozpont", "gyuru", "uresedes"]],
    left_on="join_code",
    right_on="kod",
    how="left",
    validate="many_to_one"
)

unmatched_geo = map_df["kozpont"].isna().sum()

if unmatched_geo:
    print(f"\nTérképi elemek CSV-rekord nélkül: {unmatched_geo}")

    if name_col is not None:
        print(
            map_df.loc[map_df["kozpont"].isna(), [name_col, "join_code"]]
            .head(15)
            .to_string(index=False)
        )

unmatched_csv = set(settlements["kod"]) - set(gdf["join_code"])

if unmatched_csv:
    print(f"\nCSV-s települések térképi elem nélkül: {len(unmatched_csv)}")
    names = settlements.loc[
        settlements["kod"].isin(unmatched_csv), ["kod", "telepules"]
    ]
    print(names.head(15).to_string(index=False))

map_df = map_df.dropna(subset=["kozpont"]).copy()

if map_df.empty:
    raise ValueError("Egy település sem párosítható a térképfájllal.")

print("\nTérképre került települések száma:", len(map_df))


# =========================================================
# 6. KOORDINÁTARENDSZER
# =========================================================

if map_df.crs is None:
    raise ValueError("A térképfájl koordinátarendszere nincs megadva.")

map_df = map_df.to_crs(epsg=TARGET_EPSG)


# =========================================================
# 7. SÁVOK KIOSZTÁSA
# =========================================================

values = map_df["uresedes"].to_numpy(dtype=float)
missing = np.isnan(values)

# A balról zárt intervallumok miatt a 10,0 a "10–12" sávba esik.
bin_index = np.digitize(np.nan_to_num(values, nan=-1.0), BINS)
bin_index = np.where(missing, -1, bin_index)

colors = [
    COLORS[i] if i >= 0 else NO_DATA_COLOR
    for i in bin_index
]

print("\nTelepülések száma sávonként:")
for i in range(len(BINS) + 1):
    print(f"  {bin_label(i):>10}: {(bin_index == i).sum()}")


# =========================================================
# 8. TÉRKÉP
# =========================================================

fig, ax = plt.subplots(figsize=FIGSIZE)

map_df.plot(
    ax=ax,
    color=colors,
    edgecolor="white",
    linewidth=0.08
)

# Gyűrűk határa: egy körzet egy gyűrűje egy összefüggő egység.
rings = map_df.dissolve(by=["kozpont", "gyuru"])
rings.boundary.plot(ax=ax, color="#555555", linewidth=0.35, zorder=3)

# Vonzáskörzetek külső határa vastagabban.
catchments = map_df.dissolve(by="kozpont")
catchments.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=4)


# =========================================================
# 9. KÖZPONTOK FELIRATAI
# =========================================================

texts = []

for center in sorted(map_df["kozpont"].unique()):
    center_rows = map_df[
        (map_df["kozpont"] == center) & (map_df["gyuru"] == 0)
    ]

    if center_rows.empty:
        continue

    if len(center_rows) == 1:
        point = center_rows.geometry.iloc[0].representative_point()
    else:
        point = center_rows.geometry.union_all().representative_point()

    ax.scatter(
        point.x, point.y,
        s=22, color="black", edgecolor="white", linewidth=0.6, zorder=5
    )

    texts.append(
        ax.text(
            point.x,
            point.y,
            center,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
            color="black",
            zorder=6,
            bbox=dict(
                facecolor="white",
                alpha=0.8,
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
# 10. JELMAGYARÁZAT, FORMÁZÁS ÉS MENTÉS
# =========================================================

used_classes = set(bin_index.tolist())

legend_items = [
    Patch(facecolor=COLORS[i], edgecolor="#555555", label=bin_label(i))
    for i in range(len(BINS) + 1)
    if SHOW_EMPTY_CLASSES or i in used_classes
]

if -1 in used_classes:
    legend_items.append(
        Patch(facecolor=NO_DATA_COLOR, edgecolor="#555555", label="nincs adat")
    )

# A jelmagyarázat a térképen kívül, jobb oldalon van, hogy ne takarjon ki
# semmit.
ax.legend(
    handles=legend_items,
    title="Nem lakott lakások\naránya (%)",
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    frameon=False,
    fontsize=9,
    title_fontsize=10
)

ax.set_title(
    "Szlovákia vonzáskörzeteinek gyűrűi\n"
    "nem lakott lakások aránya",
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