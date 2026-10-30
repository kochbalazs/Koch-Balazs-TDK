import re
import unicodedata
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

# A CSV és a kimeneti kép a First.py mappájában található.
BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "DATA" /"Magyarorszag" / "Alapadat" / "Huff_ábra_1.csv"
OUTPUT_PATH = BASE_DIR / "Media" /"Magyarorszag"/ "magyarorszag_vonzaskorzetek.png"

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
    """
    Településnevek egységesítése az ékezetek megtartásával.

    - A kis- és nagybetűket egységesíti.
    - Eltávolítja a név eleji és végi szóközöket.
    - Az egymást követő szóközöket egyetlen szóközre cseréli.
    - A magyar ékezeteket megtartja.
    """
    value = str(value).strip().casefold()
    value = re.sub(r"\s+", " ", value)
    return value


# =========================================================
# 3. CSV BEOLVASÁSA
# =========================================================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"A CSV-fájl nem található: {CSV_PATH}"
    )

df = pd.read_csv(
    CSV_PATH,
    encoding="cp1250",
    sep=";"
)

df.columns = df.columns.str.strip()

# A CSV oszlopneveinek egységesítése.
df = df.rename(columns={
    "Település": "telepules",
    "Központ": "kozpont",
    "település": "telepules",
    "központ": "kozpont",
})

# Ha a fájlban eleve kisbetűs oszlopnevek vannak,
# azok változtatás nélkül használhatók.
required_columns = {"telepules", "kozpont"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        "A CSV oszlopai nem megfelelőek.\n"
        f"Elvárt oszlopok: {sorted(required_columns)}\n"
        f"Talált oszlopok: {df.columns.tolist()}"
    )

df = df.dropna(subset=["telepules", "kozpont"]).copy()

df["telepules"] = df["telepules"].astype(str).str.strip()
df["kozpont"] = df["kozpont"].astype(str).str.strip()

df["join_key"] = df["telepules"].map(normalize_name)

# Üres nevek eltávolítása.
df = df[
    (df["telepules"] != "") &
    (df["kozpont"] != "") &
    (df["join_key"] != "")
].copy()

# Településszám központonként.
counts = df.groupby("kozpont")["telepules"].nunique()

print("CSV oszlopnevei:", df.columns.tolist())
print("CSV-ben szereplő települések:", df["telepules"].nunique())
print("Vonzásközpontok száma:", df["kozpont"].nunique())


# =========================================================
# 4. DUPLIKÁLT KULCSOK ELLENŐRZÉSE
# =========================================================

# Azonos normalizált névhez több vonzásközpont is tartozhat.
# Ezeket nem szabad automatikusan összevonni.

center_counts = df.groupby("join_key")["kozpont"].nunique()

conflicting_keys = center_counts[
    center_counts > 1
].index

if len(conflicting_keys) > 0:
    print("\nHIBA: Azonos névkulcshoz több vonzásközpont tartozik.")

    conflicts = (
        df[df["join_key"].isin(conflicting_keys)]
        [["telepules", "kozpont", "join_key"]]
        .sort_values(["join_key", "kozpont"])
    )

    print(conflicts.to_string(index=False))

    raise ValueError(
        "A CSV-ben ütköző településkulcsok vannak. "
        "Ezeket az adatok ellenőrzésével javítsd."
    )

# Az azonos kulcsú, azonos központhoz tartozó rekordok
# közül egyet tartunk meg az összekapcsoláshoz.
df_unique = df.drop_duplicates(
    subset="join_key",
    keep="first"
).copy()

print(
    "Egyedi településkulcsok száma:",
    df_unique["join_key"].nunique()
)


# =========================================================
# 5. TELEPÜLÉSHATÁROK LETÖLTÉSE
# =========================================================

response = requests.get(
    GEOJSON_URL,
    timeout=60
)

response.raise_for_status()

gdf = gpd.read_file(
    BytesIO(response.content)
)

print("\nGeoJSON oszlopai:")
print(gdf.columns.tolist())

# A településnév mezőjének meghatározása.
name_candidates = [
    "NAME",
    "name",
    "NAME_HU",
    "name_hu",
    "telepules",
    "Település",
    "nev",
    "Név",
    "settlement",
    "municipality",
]

name_col = next(
    (
        col for col in name_candidates
        if col in gdf.columns
    ),
    None
)

if name_col is None:
    raise ValueError(
        "Nem található a településnév oszlopa. "
        f"A GeoJSON oszlopai: {gdf.columns.tolist()}"
    )

gdf = gdf.dropna(subset=[name_col]).copy()

gdf["join_key"] = gdf[name_col].map(normalize_name)


# =========================================================
# 6. CSV ÉS TÉRKÉP ÖSSZEKAPCSOLÁSA
# =========================================================

map_df = gdf.merge(
    df_unique[["join_key", "telepules", "kozpont"]],
    on="join_key",
    how="left",
    validate="many_to_one"
)

# Azok a települések, amelyekhez nem találtunk CSV-rekordot.
missing = (
    map_df.loc[map_df["kozpont"].isna(), name_col]
    .drop_duplicates()
)

if not missing.empty:
    print("\nNem párosított települések példái:")
    print(missing.head(30).to_list())

    print(
        "Nem párosított településnevek száma:",
        len(missing)
    )

# Csak a hozzárendelt vonzásközponttal rendelkező
# települések kerülnek a térképre.
map_df = map_df.dropna(
    subset=["kozpont"]
).copy()

if map_df.empty:
    raise ValueError(
        "Egy település sem párosítható. "
        "Ellenőrizd a CSV és a GeoJSON településneveit."
    )

print(
    "\nTérképre került geometriai elemek száma:",
    len(map_df)
)

print(
    "Térképen szereplő vonzásközpontok:",
    map_df["kozpont"].nunique()
)


# =========================================================
# 7. KOORDINÁTARENDSZER
# =========================================================

if gdf.crs is None:
    raise ValueError(
        "A GeoJSON koordinátarendszere nincs megadva."
    )

# EOV vetület a magyarországi térkép megjelenítéséhez.
map_df = map_df.to_crs(epsg=23700)


# =========================================================
# 8. SZÍNEK ÉS TÉRKÉP
# =========================================================

centers = sorted(
    map_df["kozpont"].unique()
)

cmap = plt.get_cmap(
    "tab20",
    len(centers)
)

color_map = {
    center: cmap(i)
    for i, center in enumerate(centers)
}

fig, ax = plt.subplots(
    figsize=FIGSIZE
)

for center in centers:
    subset = map_df[
        map_df["kozpont"] == center
    ]

    subset.plot(
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
    subset = map_df[
        map_df["kozpont"] == center
    ]

    # A vonzásközpont saját településének geometriáját keressük.
    center_key = normalize_name(center)

    center_rows = subset[
        subset["join_key"] == center_key
    ]

    # Ha a központ nem található a saját körzetében,
    # a körzet egy reprezentatív települését használjuk.
    if center_rows.empty:
        center_rows = subset

    # A település poligonján belüli pont.
    point = (
        center_rows.geometry
        .iloc[0]
        .representative_point()
    )

    ax.scatter(
        point.x,
        point.y,
        s=10,
        color="black",
        zorder=5
    )

    label = f"{center} ({counts.get(center, 0)})"

    texts.append(
        ax.text(
            point.x,
            point.y,
            label,
            fontsize=7,
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

# Feliratok automatikus eltolása az átfedések csökkentésére.
adjust_text(
    texts,
    ax=ax,
    arrowprops=dict(
        arrowstyle="-",
        color="gray",
        lw=0.5
    )
)


# =========================================================
# 10. FORMÁZÁS ÉS MENTÉS
# =========================================================

ax.set_title(
    "Magyarország településeinek vonzáskörzetei\n"
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
