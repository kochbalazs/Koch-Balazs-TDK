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

BEOSZTAS_CSV = BASE_DIR / "DATA" / "Magyarorszag" / "gyuruk" / "Magyarország_beosztas1.csv"
URESEDES_CSV = BASE_DIR / "DATA" / "Magyarorszag" / "gyuruk" / "Magyarország_gyuruk1.csv"
OUTPUT_TEMPLATE = "magyarorszag_gyuruk_uresedes_{ev}.png"

# Mely évek térképei készüljenek el. None = mindegyik, ami a CSV-ben van
# (itt: 2022, 2011, 2001). Egy évhez: YEARS = [2022]
YEARS = None

# Eurostat GISCO LAU (települési szintű) határok. A fájl az egész EU-t
# tartalmazza (nagy), ezért az első futáskor letöltjük, kiszűrjük belőle
# Magyarországot, és a kis méretű változatot elmentjük.
GISCO_URL = (
    "https://gisco-services.ec.europa.eu/distribution/v2/lau/geojson/"
    "LAU_RG_01M_2021_4326.geojson"
)
RAW_GEO_PATH = BASE_DIR / "LAU_RG_01M_2021_4326.geojson"
CACHE_PATH = BASE_DIR / "hu_lau_telepulesek.geojson"

# Ha saját magyarországi településhatár-fájlod van (KSH-kóddal vagy
# településnévvel), add meg ide az elérési útját, és a letöltés kimarad.
LOCAL_GEO_PATH = None  # pl. BASE_DIR / "sajat_telepulesek.geojson"

# EPSG:23700 = EOV, a magyar nemzeti vetület.
TARGET_EPSG = 23700

# Hány gyűrűre osszuk a körzeteket, ha a beosztás-CSV-ben nincs gyűrű
# oszlop. Ilyenkor a nem központi településeket körzetenként a központtól
# mért távolság szerint egyenlő létszámú sávokba soroljuk (1 = legközelebbi).
# Ha a CSV-ben van gyűrűszám-oszlop (nevében "gyűrű"), azt használjuk.
RING_COUNT = 5

# Sávhatárok (%). Az intervallumok balról zártak:
# pl. a pontosan 10,0 a "10–12" sávba kerül.
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

FIGSIZE = (16, 10)
DPI = 300


# =========================================================
# 2. SEGÉDFÜGGVÉNYEK
# =========================================================

def normalize_text(value):
    """Névkulcs: ékezetek, kis/nagybetű és minden nem betű/szám nélkül."""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[^0-9a-z]", "", value.casefold())


def to_ksh(value):
    """KSH-kód egységesítése: 5 jegyű szöveg, vezető nullákkal."""
    value = re.sub(r"\.0$", "", str(value).strip())
    return value.zfill(5)


def ring_number(column_name):
    """'Központi település(ek)' -> 0, '3. gyűrű' -> 3."""
    name = str(column_name).strip()

    if "központ" in name.casefold():
        return 0

    match = re.match(r"(\d+)\.", name)

    if match is None:
        raise ValueError(f"Ismeretlen gyűrűoszlop: {column_name}")

    return int(match.group(1))


def bin_label(index):
    """A sávok felirata a jelmagyarázathoz."""
    def fmt(x):
        return f"{x:g}".replace(".", ",")

    if index == 0:
        return f"< {fmt(BINS[0])} %"

    if index == len(BINS):
        return f"≥ {fmt(BINS[-1])} %"

    return f"{fmt(BINS[index - 1])}–{fmt(BINS[index])} %"


def read_beosztas(path):
    """
    A települések beosztása: melyik körzetbe tartoznak (és ha van
    gyűrű oszlop, melyik gyűrűbe). A címsor előtti sorokat átugorjuk,
    a KSH-kódokat szövegként tartjuk.
    """
    with open(path, encoding="utf-8-sig") as f:
        lines = f.read().splitlines()

    header_idx = next(
        (i for i, line in enumerate(lines) if "KSH kód" in line),
        None
    )

    if header_idx is None:
        raise ValueError(f"A 'KSH kód' fejléc nem található: {path}")

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        sep=";",
        skiprows=header_idx,
        dtype=str
    )

    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "Település": "telepules",
        "KSH kód": "ksh",
        "Szerep": "szerep",
        "Vonzáskörzet (domináns központ)": "kozpont",
        "Központ? (1/0)": "is_kozpont",
    })

    # A gyűrűszám oszlopa: bármelyik, amelynek a nevében szerepel a "gyűrű".
    ring_cols = [
        col for col in df.columns
        if "gyűrű" in col.casefold() or "gyuru" in col.casefold()
    ]

    if ring_cols:
        df = df.rename(columns={ring_cols[0]: "gyuru"})
    else:
        print("\nA beosztás-táblában nincs gyűrű oszlop: a gyűrűket a "
              "térképi távolságból számoljuk (lásd assign_rings).")
        df["gyuru"] = np.nan

    return df


def read_vacancy_tables(path):
    """
    A gyűrűtábla több, évenként külön blokkot tartalmaz
    ("Nem lakott lakások aránya (%), 2022" stb.).
    Visszatérés: {év: hosszú alakú DataFrame (kozpont, gyuru, uresedes)}.
    """
    with open(path, encoding="utf-8-sig") as f:
        lines = f.read().splitlines()

    tables = {}
    i = 0

    while i < len(lines):
        first_cell = lines[i].split(";")[0].strip()
        year_match = re.search(r"(\d{4})\s*$", first_cell)

        if first_cell.casefold().startswith("nem lakott") and year_match:
            year = int(year_match.group(1))
            header = [c.strip() for c in lines[i + 1].split(";")]

            rows = []
            j = i + 2

            while j < len(lines) and lines[j].split(";")[0].strip() != "":
                rows.append([c.strip() for c in lines[j].split(";")])
                j += 1

            wide = pd.DataFrame(rows, columns=header)

            center_col = header[0]

            # Az összesítő sor nem körzet.
            wide = wide[
                ~wide[center_col].str.casefold().str.startswith("összes")
            ].copy()

            value_cols = header[1:]

            long = wide.melt(
                id_vars=[center_col],
                value_vars=value_cols,
                var_name="oszlop",
                value_name="uresedes"
            )

            long = long.rename(columns={center_col: "kozpont"})
            long["gyuru"] = long["oszlop"].map(ring_number)
            long["uresedes"] = pd.to_numeric(
                long["uresedes"].str.replace(",", ".", regex=False),
                errors="coerce"
            )

            tables[year] = long[["kozpont", "gyuru", "uresedes"]]
            i = j
        else:
            i += 1

    if not tables:
        raise ValueError(f"Nem található üresedési blokk a fájlban: {path}")

    return tables


def load_settlements():
    """Magyarországi településhatárok betöltése (gyorsítótárral)."""
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

    hu = raw[raw["CNTR_CODE"] == "HU"].copy()

    if hu.empty:
        raise ValueError("A GISCO-fájlban nem található magyar település.")

    hu.to_file(CACHE_PATH, driver="GeoJSON")
    print(f"Magyarországi gyorsítótár elmentve: {CACHE_PATH}")

    return hu


def assign_rings(df, n_rings=RING_COUNT):
    """
    A nem központi településeket körzetenként n_rings egyenlő létszámú
    gyűrűbe sorolja a központtól (a Balatonnál: a legközelebbi parti
    településtől) mért távolság szerint. 1 = legközelebbi gyűrű.
    A df vetített (méteres) koordinátarendszerben legyen.
    """
    df = df.copy()
    centroids = df.geometry.centroid

    for center, grp in df.groupby("kozpont"):
        centers = grp[grp["gyuru"] == 0]
        others = grp[grp["gyuru"] != 0]

        if centers.empty or others.empty:
            continue

        center_pts = centroids.loc[centers.index].union_all()
        dist = centroids.loc[others.index].distance(center_pts)

        ranks = dist.rank(method="first")
        df.loc[others.index, "gyuru"] = np.ceil(ranks / len(dist) * n_rings)

    return df


# =========================================================
# 3. TÁBLÁZATOK BEOLVASÁSA
# =========================================================

for path in (BEOSZTAS_CSV, URESEDES_CSV):
    if not path.exists():
        raise FileNotFoundError(f"A CSV-fájl nem található: {path}")

settlements = read_beosztas(BEOSZTAS_CSV)

required = {"telepules", "ksh", "kozpont", "is_kozpont", "gyuru"}

if not required.issubset(settlements.columns):
    raise ValueError(
        "A beosztás-tábla oszlopai nem megfelelőek.\n"
        f"Elvárt: {sorted(required)}\n"
        f"Talált: {settlements.columns.tolist()}"
    )

settlements = settlements.dropna(subset=["ksh", "kozpont"]).copy()

settlements["ksh"] = settlements["ksh"].map(to_ksh)
settlements["telepules"] = settlements["telepules"].astype(str).str.strip()
settlements["kozpont"] = settlements["kozpont"].astype(str).str.strip()

settlements["is_kozpont"] = pd.to_numeric(
    settlements["is_kozpont"], errors="coerce"
).fillna(0).astype(int)

settlements["gyuru"] = pd.to_numeric(settlements["gyuru"], errors="coerce")

# A központi települések (a Balaton-parti települések is) a 0. gyűrűbe
# tartoznak, akkor is, ha a gyűrűoszlop náluk üres.
settlements.loc[settlements["is_kozpont"] == 1, "gyuru"] = 0

# A nem központi települések gyűrűszámát (ha a CSV-ben nincs) a térkép
# betöltése után számoljuk (6. szakasz).
settlements["gyuru"] = settlements["gyuru"].astype(float)

vacancy_by_year = read_vacancy_tables(URESEDES_CSV)

years = sorted(vacancy_by_year.keys(), reverse=True)

if YEARS is not None:
    years = [y for y in years if y in YEARS]

print("Települések száma a CSV-ben:", settlements["ksh"].nunique())
print("Vonzáskörzetek száma:", settlements["kozpont"].nunique())
print("Üresedési táblák évei:", sorted(vacancy_by_year.keys()))


# =========================================================
# 4. ELLENŐRZÉSEK
# =========================================================

if settlements["ksh"].duplicated().any():
    dupes = settlements.loc[settlements["ksh"].duplicated(keep=False), "ksh"]
    raise ValueError(
        "Ismétlődő KSH-kódok a beosztás-táblában: "
        f"{sorted(dupes.unique())[:20]}"
    )

for year, vacancy in vacancy_by_year.items():
    only_in_settlements = set(settlements["kozpont"]) - set(vacancy["kozpont"])
    only_in_vacancy = set(vacancy["kozpont"]) - set(settlements["kozpont"])

    if only_in_settlements or only_in_vacancy:
        print(f"\nFIGYELEM ({year}): a körzetnevek eltérnek a két táblában.")
        print("Csak a beosztás-táblában:", sorted(only_in_settlements))
        print("Csak az üresedési táblában:", sorted(only_in_vacancy))


# =========================================================
# 5. TELEPÜLÉSHATÁROK BETÖLTÉSE ÉS ÖSSZEKAPCSOLÁS
# =========================================================

gdf = load_settlements()

print("\nGeoJSON oszlopai:")
print(gdf.columns.tolist())

# 5/a. A térképfájl kód- és névoszlopa.
code_candidates = ["LAU_ID", "LAU_CODE", "KSH", "ksh", "KOD", "kod", "id"]
name_candidates = ["LAU_NAME", "NAME", "name", "NAME_HU", "name_hu",
                   "telepules", "Település", "nev", "Név"]

code_col = next((c for c in code_candidates if c in gdf.columns), None)
name_col = next((c for c in name_candidates if c in gdf.columns), None)

if code_col is not None:
    gdf["join_code"] = gdf[code_col].map(to_ksh)
elif "GISCO_ID" in gdf.columns:
    # A GISCO_ID formátuma: "HU_13578".
    gdf["join_code"] = gdf["GISCO_ID"].astype(str).str.split("_").str[-1].map(to_ksh)
    code_col = "GISCO_ID"
else:
    gdf["join_code"] = None

if name_col is not None:
    gdf["join_name"] = gdf[name_col].map(normalize_text)
else:
    gdf["join_name"] = None

settlements["join_name"] = settlements["telepules"].map(normalize_text)

# 5/b. A jobb összekapcsolási mód kiválasztása: KSH-kód vagy név.
csv_codes = set(settlements["ksh"])
csv_names = set(settlements["join_name"])

code_cov = (
    len(csv_codes & set(gdf["join_code"].dropna())) / len(csv_codes)
    if code_col is not None else 0.0
)
name_cov = (
    len(csv_names & set(gdf["join_name"].dropna())) / len(csv_names)
    if name_col is not None else 0.0
)

print(f"\nKód alapú lefedettség: {code_cov:.1%} (oszlop: {code_col})")
print(f"Név alapú lefedettség: {name_cov:.1%} (oszlop: {name_col})")

if max(code_cov, name_cov) == 0:
    raise ValueError(
        "Egy település sem párosítható sem kód, sem név alapján. "
        f"A GeoJSON oszlopai: {gdf.columns.tolist()}"
    )

if code_cov >= name_cov:
    join_left, join_right, join_kind = "join_code", "ksh", "KSH-kód"
else:
    join_left, join_right, join_kind = "join_name", "join_name", "név"

print("Összekapcsolás módja:", join_kind)

if join_kind == "név" and settlements["join_name"].duplicated().any():
    raise ValueError("A névkulcsok nem egyediek, név alapon nem kapcsolható.")

map_df = gdf.merge(
    settlements[["ksh", "join_name", "telepules", "kozpont", "gyuru"]],
    left_on=join_left,
    right_on=join_right,
    how="left",
    validate="many_to_one",
    suffixes=("", "_csv")
)

unmatched_geo = map_df["kozpont"].isna().sum()

if unmatched_geo:
    print(f"\nTérképi elemek CSV-rekord nélkül: {unmatched_geo}")

matched_csv = set(map_df["ksh"].dropna())
unmatched_csv = csv_codes - matched_csv

if unmatched_csv:
    print(f"CSV-s települések térképi elem nélkül: {len(unmatched_csv)}")
    names = settlements.loc[
        settlements["ksh"].isin(unmatched_csv), "telepules"
    ]
    print("Példák:", names.head(20).to_list())

map_df = map_df.dropna(subset=["kozpont"]).copy()

if map_df.empty:
    raise ValueError("Egy település sem párosítható a térképfájllal.")

print("\nTérképre került települések száma:", len(map_df))


# =========================================================
# 6. KOORDINÁTARENDSZER ÉS GYŰRŰK
# =========================================================

if map_df.crs is None:
    raise ValueError("A térképfájl koordinátarendszere nincs megadva.")

map_df = map_df.to_crs(epsg=TARGET_EPSG).reset_index(drop=True)
map_df["gyuru"] = map_df["gyuru"].astype(float)

# Ha a CSV-ben nem volt gyűrűszám, a központtól mért távolságból számoljuk.
if map_df.loc[map_df["gyuru"] != 0, "gyuru"].isna().any():
    map_df = assign_rings(map_df)
    print("\nGyűrűk távolság alapján számolva, települések száma gyűrűnként:")
    print(map_df["gyuru"].value_counts(dropna=False).sort_index().to_string())

map_df["gyuru"] = map_df["gyuru"].astype("Int64")


# =========================================================
# 7. ÉVTŐL FÜGGETLEN RÉTEGEK
# =========================================================

# Gyűrűk határa: egy körzet egy gyűrűje egy összefüggő egység.
rings = map_df.dropna(subset=["gyuru"]).dissolve(by=["kozpont", "gyuru"])

# Vonzáskörzetek külső határa.
catchments = map_df.dissolve(by="kozpont")

# Központok jelölőpontja és felirata.
center_points = []

for center in sorted(map_df["kozpont"].unique()):
    center_rows = map_df[
        (map_df["kozpont"] == center) & (map_df["gyuru"] == 0)
    ]

    if center_rows.empty:
        continue

    exact = center_rows[
        center_rows["telepules"].map(normalize_text) == normalize_text(center)
    ]

    if len(exact) == 1:
        point = exact.geometry.iloc[0].representative_point()
    else:
        # Több központi település (pl. Balaton): a csoport súlypontja.
        point = center_rows.geometry.union_all().centroid

    center_points.append((center, point))


# =========================================================
# 8. TÉRKÉP RAJZOLÁSA
# =========================================================

def draw_map(year, vacancy):
    # Az adott év üresedési értéke minden településhez a saját
    # körzetének és gyűrűjének értéke. (A bal oldali sorrend megmarad.)
    values = (
        map_df[["kozpont", "gyuru"]]
        .astype({"gyuru": "float"})
        .merge(
            vacancy.astype({"gyuru": "float"}),
            on=["kozpont", "gyuru"],
            how="left"
        )["uresedes"]
        .to_numpy(dtype=float)
    )

    missing = np.isnan(values)

    # A balról zárt intervallumok miatt a 10,0 a "10–12" sávba esik.
    bin_index = np.digitize(np.nan_to_num(values, nan=-1.0), BINS)
    bin_index = np.where(missing, -1, bin_index)

    colors = [
        COLORS[i] if i >= 0 else NO_DATA_COLOR
        for i in bin_index
    ]

    print(f"\n{year} - települések száma sávonként:")
    for i in range(len(BINS) + 1):
        print(f"  {bin_label(i):>10}: {(bin_index == i).sum()}")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    map_df.plot(
        ax=ax,
        color=colors,
        edgecolor="white",
        linewidth=0.08
    )

    rings.boundary.plot(ax=ax, color="#555555", linewidth=0.35, zorder=3)
    catchments.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=4)

    texts = []

    for center, point in center_points:
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

    used_classes = set(bin_index.tolist())

    legend_items = [
        Patch(facecolor=COLORS[i], edgecolor="#555555", label=bin_label(i))
        for i in range(len(BINS) + 1)
        if SHOW_EMPTY_CLASSES or i in used_classes
    ]

    if -1 in used_classes:
        legend_items.append(
            Patch(facecolor=NO_DATA_COLOR, edgecolor="#555555",
                  label="nincs adat")
        )

    # A jelmagyarázat a térképen kívül, jobb oldalon van, hogy ne
    # takarjon ki semmit.
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
        "Magyarország vonzáskörzeteinek gyűrűi\n"
        f"nem lakott lakások aránya, {year}",
        fontsize=16,
        fontweight="bold",
        loc="left",
        pad=15
    )

    ax.axis("off")
    ax.set_aspect("equal")

    plt.tight_layout()

    output_path = BASE_DIR / OUTPUT_TEMPLATE.format(ev=year)

    plt.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white"
    )

    print(f"A térkép elkészült: {output_path}")


# =========================================================
# 9. TÉRKÉPEK ELKÉSZÍTÉSE ÉVENKÉNT
# =========================================================

for year in years:
    draw_map(year, vacancy_by_year[year])

plt.show()