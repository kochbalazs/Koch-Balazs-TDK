import re
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

GMINA_CSV = BASE_DIR / "DATA" / "Lengyelorszag" / "Lengyelország_ábra.csv"
URESEDES_CSV = BASE_DIR / "DATA" / "Lengyelorszag" / "Lengyelország_gyurukcsv.csv"
OUTPUT_PATH = BASE_DIR / "Media" / "Lengyelorszag" / "lengyelorszag_gyuruk_uresedes.png"

# Ha saját gminahatár-fájlod van (shp / geojson / gpkg), add meg itt.
LOCAL_GEO_PATH = None  # pl. BASE_DIR / "DATA" / "Lengyelorszag" / "gminy.shp"

# 1. forrás: lengyel gminahatárok 7 jegyű TERC-kóddal ("terc" tulajdonság).
PRG_URLS = [
    "https://raw.githubusercontent.com/jusuff/PolandGeoJson/"
    "main/data/poland.municipalities.json",
    "https://raw.githubusercontent.com/jusuff/PolandGeoJson/"
    "main/data/poland.municipalities.geojson",
]
PRG_CACHE = BASE_DIR / "pl_gminy_prg.geojson"

# 2. forrás (tartalék): Eurostat GISCO LAU, csak Lengyelországra szűrve.
GISCO_URL = (
    "https://gisco-services.ec.europa.eu/distribution/v2/lau/geojson/"
    "LAU_RG_01M_2021_4326.geojson"
)
GISCO_RAW = BASE_DIR / "LAU_RG_01M_2021_4326.geojson"
GISCO_CACHE = BASE_DIR / "pl_gminy.geojson"

# Ha egy forrás a CSV-kódok ennél kisebb hányadát találja meg,
# megpróbáljuk a következő forrást.
MIN_MATCH_RATIO = 0.95

# EPSG:2180 = PL-1992, a lengyel nemzeti vetület.
TARGET_EPSG = 2180

# Üresedési sávhatárok (%). Balról zárt intervallumok:
# a pontosan 10,0 a "10–12" sávba kerül.
BINS = [8, 10, 12, 15, 18, 21, 25]

# ColorBrewer YlOrRd, 8 osztály: minél nagyobb az üresedés, annál sötétebb.
COLORS = [
    "#ffffcc", "#ffeda0", "#fed976", "#feb24c",
    "#fd8d3c", "#fc4e2a", "#e31a1c", "#b10026",
]

NO_DATA_COLOR = "#d9d9d9"

# True: a jelmagyarázatban minden sáv szerepel, akkor is, ha üres.
SHOW_EMPTY_CLASSES = True

FIGSIZE = (14, 12)
DPI = 300

# Ezekben az oszlopokban keressük a gmina kódját a térképfájlban.
CODE_CANDIDATES = [
    "terc", "TERC", "JPT_KOD_JE", "LAU_ID", "LAU_CODE", "TERYT", "teryt",
    "KOD", "kod", "id", "code", "GISCO_ID",
]


# =========================================================
# 2. SEGÉDFÜGGVÉNYEK
# =========================================================

def read_table(path, key):
    """
    CSV beolvasása: a fejléc az első olyan sor, amelyben a `key` szerepel,
    az előtte lévő (cím)sorokat átugorjuk. A TERYT-kódok szövegek maradnak
    (vezető nullák!), a tizedesvessző kezelve van.
    """
    with open(path, encoding="utf-8-sig") as f:
        lines = f.read().splitlines()

    header_idx = next(
        (i for i, line in enumerate(lines) if key in line),
        None
    )

    if header_idx is None:
        raise ValueError(f"A(z) '{key}' fejléc nem található: {path}")

    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        sep=";",
        skiprows=header_idx,
        decimal=",",
        dtype={"TERYT (7 jegy)": str, "TERYT (6 jegy)": str}
    )


def ring_number(column_name):
    """'Központ' -> 0, '3. gyűrű' -> 3."""
    name = str(column_name).strip()

    if name.casefold() == "központ":
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


def download(url, target):
    """Fájl letöltése lemezre, darabokban."""
    print(f"Letöltés: {url}")
    with requests.get(
        url,
        stream=True,
        timeout=120,
        headers={"User-Agent": "Mozilla/5.0"}
    ) as response:
        response.raise_for_status()
        with open(target, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)


def load_prg():
    """Lengyel gminahatárok 7 jegyű TERC-kóddal (gyorsítótárral)."""
    if PRG_CACHE.exists():
        return gpd.read_file(PRG_CACHE)

    for url in PRG_URLS:
        try:
            download(url, PRG_CACHE)
            return gpd.read_file(PRG_CACHE)
        except Exception as error:
            print(f"Nem sikerült ({error}), következő cím...")
            if PRG_CACHE.exists():
                PRG_CACHE.unlink()

    return None


def load_gisco():
    """GISCO LAU, csak Lengyelország (gyorsítótárral)."""
    if GISCO_CACHE.exists():
        return gpd.read_file(GISCO_CACHE)

    try:
        if not GISCO_RAW.exists():
            download(GISCO_URL, GISCO_RAW)

        raw = gpd.read_file(GISCO_RAW)
    except Exception as error:
        print(f"A GISCO-fájl nem érhető el ({error}).")
        return None

    pl = raw[raw["CNTR_CODE"] == "PL"].copy()

    if pl.empty:
        return None

    pl.to_file(GISCO_CACHE, driver="GeoJSON")
    return pl


def code_variants(series):
    """
    A térképi kódoszlop többféle lehetséges átalakítása 6 jegyű kóddá.
    Azért kell több, mert előre nem tudjuk, hogy a kód 7 jegyű-e,
    elveszett-e a vezető nulla, vagy van-e előtag.
    """
    digits = (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
    )

    return {
        "7 jegyre kiegészítve, első 6 jegy": digits.str.zfill(7).str[:6],
        "első 6 jegy": digits.str[:6].str.zfill(6),
        "utolsó 6 jegy": digits.str[-6:].str.zfill(6),
        "6 jegyre kiegészítve": digits.str.zfill(6),
    }


def find_best_key(gdf, csv_codes):
    """
    Végigpróbálja a lehetséges kódoszlopokat és átalakításokat, és azt
    választja, amelyik a legtöbb CSV-kódot találja meg.
    Visszatér: (találatok, oszlop, átalakítás, 6 jegyű kódok)
    """
    best = (0, None, None, None)

    for col in CODE_CANDIDATES:
        if col not in gdf.columns:
            continue

        raw = gdf[col]

        if col == "GISCO_ID":
            raw = raw.astype(str).str.split("_").str[-1]

        for name, codes in code_variants(raw).items():
            score = len(set(codes) & csv_codes)

            if score > best[0]:
                best = (score, col, name, codes)

    return best


def pick_geometry(csv_codes):
    """
    Forrásonként végigpróbálja a térképfájlokat, és az első olyat
    használja, amely a CSV-kódok MIN_MATCH_RATIO-nyi részét megtalálja.
    Ha egyik sem éri el, a legjobbat adja vissza.
    """
    loaders = []

    if LOCAL_GEO_PATH is not None:
        loaders.append(("saját fájl", lambda: gpd.read_file(LOCAL_GEO_PATH)))
    else:
        loaders.append(("PRG (TERC)", load_prg))
        loaders.append(("GISCO LAU", load_gisco))

    results = []

    for name, loader in loaders:
        gdf = loader()

        if gdf is None:
            print(f"Forrás kihagyva: {name}")
            continue

        key = find_best_key(gdf, csv_codes)
        print(f"Forrás: {name}: {key[0]}/{len(csv_codes)} kód párosítható")
        results.append((gdf, key, name))

        if key[0] >= MIN_MATCH_RATIO * len(csv_codes):
            return results[-1]

    if not results:
        raise RuntimeError(
            "Nem sikerült gminahatár-fájlt betölteni. Add meg a "
            "LOCAL_GEO_PATH-ban egy saját fájl elérési útját."
        )

    return max(results, key=lambda r: r[1][0])


# =========================================================
# 3. TÁBLÁZATOK BEOLVASÁSA
# =========================================================

for path in (GMINA_CSV, URESEDES_CSV):
    if not path.exists():
        raise FileNotFoundError(f"A CSV-fájl nem található: {path}")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 3/a. Gminák: melyik központ körzetébe és melyik gyűrűbe tartoznak.
gminas = read_table(GMINA_CSV, "TERYT")

gminas = gminas.rename(columns={
    "TERYT (7 jegy)": "teryt7",
    "TERYT (6 jegy)": "teryt6",
    "Gmina": "gmina",
    "Vonzáskörzet": "kozpont",
    "Központ? (1/0)": "is_kozpont",
    "Gyűrű (0 = központ)": "gyuru",
})

required = {"teryt7", "teryt6", "gmina", "kozpont", "is_kozpont", "gyuru"}

if not required.issubset(gminas.columns):
    raise ValueError(
        "A gmina-tábla oszlopai nem megfelelőek.\n"
        f"Elvárt: {sorted(required)}\n"
        f"Talált: {gminas.columns.tolist()}"
    )

gminas = gminas.dropna(subset=["teryt6", "kozpont", "gyuru"]).copy()
gminas["teryt7"] = gminas["teryt7"].astype(str).str.strip().str.zfill(7)
gminas["teryt6"] = gminas["teryt6"].astype(str).str.strip().str.zfill(6)
gminas["kozpont"] = gminas["kozpont"].astype(str).str.strip()
gminas["gyuru"] = gminas["gyuru"].astype(int)
gminas["is_kozpont"] = gminas["is_kozpont"].astype(int)

# 3/b. Üresedés központonként és gyűrűnként (széles -> hosszú alak).
vacancy_wide = read_table(URESEDES_CSV, "Vonzáskörzet (központ)")

center_col = vacancy_wide.columns[0]
vacancy_wide[center_col] = vacancy_wide[center_col].astype(str).str.strip()

# Az összesítő sor nem központ.
vacancy_wide = vacancy_wide[
    ~vacancy_wide[center_col].str.casefold().str.startswith("összes")
].copy()

ring_columns = [
    col for col in vacancy_wide.columns
    if str(col).strip().casefold() == "központ"
    or re.match(r"\d+\.", str(col).strip())
]

vacancy = vacancy_wide.melt(
    id_vars=[center_col],
    value_vars=ring_columns,
    var_name="oszlop",
    value_name="uresedes"
)

vacancy = vacancy.rename(columns={center_col: "kozpont"})
vacancy["gyuru"] = vacancy["oszlop"].map(ring_number)
vacancy["uresedes"] = pd.to_numeric(vacancy["uresedes"], errors="coerce")

print("Gminák száma a CSV-ben:", gminas["teryt6"].nunique())
print("Központok száma:", gminas["kozpont"].nunique())
print("Üresedési értékek száma:", vacancy["uresedes"].notna().sum())


# =========================================================
# 4. ELLENŐRZÉSEK
# =========================================================

if gminas["teryt6"].duplicated().any():
    dupes = gminas.loc[gminas["teryt6"].duplicated(keep=False), "teryt6"]
    raise ValueError(
        "Ismétlődő TERYT-kódok a gmina-táblában: "
        f"{sorted(dupes.unique())[:20]}"
    )

# A központnevek egyezzenek a két táblában.
only_in_gminas = set(gminas["kozpont"]) - set(vacancy["kozpont"])
only_in_vacancy = set(vacancy["kozpont"]) - set(gminas["kozpont"])

if only_in_gminas or only_in_vacancy:
    print("\nFIGYELEM: központnevek eltérnek a két táblában.")
    print("Csak a gmina-táblában:", sorted(only_in_gminas))
    print("Csak az üresedési táblában:", sorted(only_in_vacancy))

# Az üresedési tábla "Nem központi gminák (n)" oszlopa ellenőrzésre jó.
n_col = next(
    (c for c in vacancy_wide.columns if "Nem központi" in str(c)),
    None
)

if n_col is not None:
    expected = vacancy_wide.set_index(center_col)[n_col].astype(int)
    actual = gminas[gminas["is_kozpont"] == 0].groupby("kozpont").size()
    diff = (actual.reindex(expected.index).fillna(0).astype(int) - expected)
    diff = diff[diff != 0]

    if len(diff):
        print("\nFIGYELEM: a nem központi gminák száma eltér:")
        print(diff.to_string())

data = gminas.merge(
    vacancy[["kozpont", "gyuru", "uresedes"]],
    on=["kozpont", "gyuru"],
    how="left",
    validate="many_to_one"
)

n_missing = data["uresedes"].isna().sum()

if n_missing > 0:
    print(f"\nFIGYELEM: {n_missing} gminához nincs üresedési érték:")
    print(
        data.loc[data["uresedes"].isna(), ["gmina", "kozpont", "gyuru"]]
        .drop_duplicates(["kozpont", "gyuru"])
        .to_string(index=False)
    )


# =========================================================
# 5. SÁVOK KIOSZTÁSA
# =========================================================

bin_index = np.digitize(data["uresedes"].fillna(-1).to_numpy(), BINS)

data["sav"] = np.where(data["uresedes"].isna(), -1, bin_index)

data["szin"] = [
    COLORS[i] if i >= 0 else NO_DATA_COLOR
    for i in data["sav"]
]

print("\nGminák száma sávonként:")
for i in range(len(BINS) + 1):
    print(f"  {bin_label(i):>10}: {(data['sav'] == i).sum()}")


# =========================================================
# 6. GMINAHATÁROK BETÖLTÉSE ÉS ÖSSZEKAPCSOLÁS
# =========================================================

csv_codes = set(data["teryt6"])

gdf, (score, code_col, variant, codes6), source_name = pick_geometry(csv_codes)

print(f"\nHasznált térképforrás: {source_name}")
print("Térkép oszlopai:", gdf.columns.tolist())

if code_col is None or score == 0:
    print("\nEgyetlen kódoszlop sem párosítható. Minták a térképfájlból:")
    for col in CODE_CANDIDATES:
        if col in gdf.columns:
            print(f"  {col}: {gdf[col].head(8).tolist()}")
    print("Minták a CSV-ből:", sorted(csv_codes)[:8])
    raise ValueError(
        "Egy gmina sem párosítható. Ellenőrizd a fenti mintákat, "
        "vagy add meg a saját fájlt a LOCAL_GEO_PATH-ban."
    )

print(
    f"Használt kódoszlop: {code_col} (átalakítás: {variant}), "
    f"párosított kódok: {score}/{len(csv_codes)}"
)

gdf = gdf.copy()
gdf["teryt6"] = codes6

merged = gdf.merge(
    data[["teryt6", "gmina", "kozpont", "gyuru", "uresedes", "sav", "szin"]],
    on="teryt6",
    how="left",
    validate="many_to_one"
)

unmatched_geo = int(merged["kozpont"].isna().sum())
unmatched_csv = csv_codes - set(gdf["teryt6"])

if unmatched_geo:
    print(f"\nTérképi elemek CSV-rekord nélkül: {unmatched_geo} (szürkén jelennek meg)")

if unmatched_csv:
    print(f"CSV-s kódok térképi elem nélkül: {len(unmatched_csv)}")
    print("Példák:", sorted(unmatched_csv)[:20])

# A párosítatlan elemek szürkék; a határok és feliratok csak a párosítottakból épülnek.
merged["szin"] = merged["szin"].fillna(NO_DATA_COLOR)
merged["sav"] = merged["sav"].fillna(-1).astype(int)

map_df = merged.dropna(subset=["kozpont"]).copy()

if map_df.empty:
    raise ValueError("Egy gmina sem párosítható.")

print("\nTérképre került gminák száma:", len(map_df))


# =========================================================
# 7. KOORDINÁTARENDSZER
# =========================================================

if merged.crs is None:
    raise ValueError("A térképfájl koordinátarendszere nincs megadva.")

merged = merged.to_crs(epsg=TARGET_EPSG)
map_df = map_df.to_crs(epsg=TARGET_EPSG)


# =========================================================
# 8. TÉRKÉP
# =========================================================

fig, ax = plt.subplots(figsize=FIGSIZE)

# Minden gmina (a párosítatlanok szürkén) a saját sávszínével.
merged.plot(
    ax=ax,
    color=merged["szin"],
    edgecolor="white",
    linewidth=0.08
)

# Gyűrűk határa: egy központ egy gyűrűje egy összefüggő egység.
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

    point = center_rows.geometry.iloc[0].representative_point()

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

used_classes = set(merged["sav"].astype(int).unique())

legend_items = [
    Patch(facecolor=COLORS[i], edgecolor="#555555", label=bin_label(i))
    for i in range(len(BINS) + 1)
    if SHOW_EMPTY_CLASSES or i in used_classes
]

if (merged["sav"] == -1).any():
    legend_items.append(
        Patch(facecolor=NO_DATA_COLOR, edgecolor="#555555", label="nincs adat")
    )

ax.legend(
    handles=legend_items,
    title="Üresedés (%)",
    loc="lower left",
    frameon=True,
    framealpha=0.9,
    fontsize=9,
    title_fontsize=10
)

ax.set_title(
    "Lengyelország vonzáskörzeteinek gyűrűi\n"
    "lakásszámmal súlyozott üresedés szerint",
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