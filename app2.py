import streamlit as st                                                  # Streamlit könyvtár a gyors webes UI-hoz
import boto3                                                            # AWS SDK Pythonhoz, a felhőszolgáltatások kezeléséhez
import pandas as pd                                                     # Adatkezeléshez és vizualizációhoz
import matplotlib.pyplot as plt                                         # Grafikonok rajzolásához
from botocore.exceptions import ClientError                             # AWS hibakezeléshez
from decimal import Decimal                                             # Pontos számábrázoláshoz, különösen pénzügyi adatoknál
import uuid                                                             # Egyedi azonosítók generálásához (pl. termék ID-k)
import time                                                             # Időzítéshez, késleltetés szimulálásához (pl. Káosz Kísérlet)

# AWS KONFIGURÁCIÓ
# Ezek az azonosítók kötik össze a kódot az infrastruktúrával a felhőben
REGION = 'eu-central-1'                                                 # AWS régió, ahol a szolgáltatások futnak
TABLE_NAME = 'Inventory'                                           # DynamoDB tábla neve, ahol a készletadatok tárolódnak
USER_POOL_ID = 'eu-central-1_nj6txDNAV'                                 # AWS Cognito User Pool azonosító
APP_CLIENT_ID = '4fqhvpc1ajg4rnj2ka3jns2pmp'                            # AWS Cognito App Client ID (SPA mód)

# AWS Szolgáltatások inicializálása a Boto3 SDK segítségével
client = boto3.client('cognito-idp', region_name=REGION)                # Hitelesítésért felelős kliens
dynamodb = boto3.resource('dynamodb', region_name=REGION)               # Adatbázis erőforrás
table = dynamodb.Table(TABLE_NAME)                                      # DynamoDB tábla objektum a CRUD műveletekhez

#  BACKEND / ADATKEZELŐ FÜGGVÉNYEK 

def login_user(username, password)-> dict:
    """
    Hitelesítési folyamat (Authentication Flow).
    Közvetlenül az AWS Cognito-val kommunikál USER_PASSWORD_AUTH módban.
    """
    try:                                                                # A Cognito initiate_auth hívása a megadott hitelesítési paraméterekkel
        response = client.initiate_auth(
            ClientId=APP_CLIENT_ID,
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters={'USERNAME': username, 'PASSWORD': password}
        )
        return response                                                # Sikeres hitelesítés esetén JWT(JSON Web Token) tokeneket kapunk
    except ClientError as e:                                           # Hibakezelés: ha a hitelesítés sikertelen, a Cognito egy ClientError-t dob, amit itt kezelünk
        st.error(f"Hitelesítési hiba: {e.response['Error']['Message']}")
        return None                                                    # Sikertelen hitelesítés esetén None-t adunk vissza, ami a frontend logikában kezelve lesz

def get_data(simulate_chaos=False)-> list:
    """
    Adatlekérés a DynamoDB-ből (Read művelet).
    Tartalmaz egy kapcsolót a Káosz Kísérlet szimulálásához.
    """
    if simulate_chaos:                                                 # Ha a Káosz Kísérlet szimulációja engedélyezve van, akkor szándékosan késleltetjük a választ és hibát dobunk
        time.sleep(2)
        raise Exception("Szimulált hálózati hiba a DynamoDB elérésekor!")
    
    # Scan művelet: lekéri az összes tételt a táblából
    response = table.scan()
    return response.get('Items', [])                                    # Visszaadja a lekért tételeket, vagy egy üres listát, ha nincs adat

def add_new_product(warehouse_id, name, qty, cat)-> None:
    """
    Új tétel rögzítése (Create művelet).
    UUID-t használunk az egyedi azonosításhoz, Decimal-t a mennyiséghez.
    """
    prod_id = str(uuid.uuid4())[:8]                                     # Rövid egyedi azonosító generálása a termékhez 
    try:
        table.put_item(
            Item={
                'PK': f"WH#{warehouse_id}",                             # Partíciós kulcs: raktár azonosítója
                'SK': f"PROD#{prod_id}",                                # Rendezési kulcs: termék azonosítója
                'product_name': name,                                   # Egyéb attribútumok: termék neve, mennyiség és kategória
                'quantity': Decimal(str(qty)),                          # Decimal típus a kerekítési hibák ellen
                'category': cat                                         # Kategória, pl. Elektronika, Szerszámok, stb.
            }
        )
        st.success(f"Termék rögzítve: {name}")                          # Sikerüzenet megjelenítése a felhasználónak
        time.sleep(1)                                                   # Rövid szünet a sikerüzenet megjelenítéséhez/adatbázis frissítéséhez
        st.rerun()                                                      # Oldal frissítése az új adatok megjelenítéséhez
    except Exception as e:
        st.error(f"Adatbázis írási hiba: {e}")                          # Hibakezelés: ha a put_item művelet sikertelen, a kivétel itt lesz elkapva és egy hibaüzenet jelenik meg a felhasználónak

def update_quantity(warehouse_id, product_id, new_qty)-> None:
    """
    Készletmódosítás (Update művelet).
    Optimális frissítés: csak a 'quantity' attribútum változik a megadott kulcsok alapján.
    """
    try:
        table.update_item(
            Key={'PK': f"WH#{warehouse_id}", 'SK': f"PROD#{product_id}"},
            UpdateExpression="set quantity = :q",
            ExpressionAttributeValues={':q': Decimal(str(new_qty))}
        )                                                               # Update művelet végrehajtása a DynamoDB-ben a megadott kulcsok alapján, csak a 'quantity' mező frissítése
        st.success("Mennyiség frissítve!")                              # Sikerüzenet megjelenítése a felhasználónak
        time.sleep(1)
        st.rerun()
    except Exception as e:
        st.error(f"Frissítési hiba: {e}")

#  FRONTEND / FELHASZNÁLÓI FELÜLET (UI) 

# Session State inicializálása a bejelentkezési állapot megőrzéséhez
if "authenticated" not in st.session_state:                             # Alapértelmezett érték: nem hitelesített állapot
    st.session_state["authenticated"] = False

# 1. Bejelentkezési képernyő (Autentikációs réteg)
if not st.session_state["authenticated"]:                               # Ha a felhasználó nincs hitelesítve, megjelenítjük a bejelentkezési űrlapot
    st.title("🔐 Raktárkezelő Rendszer - Belépés")                      # Címsor a bejelentkezési oldal tetején
    st.write("Jelentkezz be a hozzáféréshez!")                          # Rövid instrukció a felhasználónak
    user = st.text_input("Felhasználónév")                              # Szövegmező a felhasználónév megadásához
    pw = st.text_input("Jelszó", type="password")                       # Szövegmező a jelszó megadásához, karakterek elrejtése
    if st.button("Bejelentkezés"):                                      # Bejelentkezés gomb, amire kattintva megpróbáljuk hitelesíteni a felhasználót
        if login_user(user, pw):                                        # Ha a hitelesítés sikeres, a session_state-ben beállítjuk az authenticated értékét True-ra, és újratöltjük az oldalt
            st.session_state["authenticated"] = True
            st.rerun()

# 2. Védett tartalom (Dashboard és Admin funkciók)
else:
    st.title("📊 Raktárkészlet Vizualizáció")
    
    # Kijelentkezés és Káosz teszt a sidebar-on
    if st.sidebar.button("Kijelentkezés"):                              # Kijelentkezés gomb a sidebar-on, amire kattintva visszaállítjuk az authenticated értékét False-ra, és újratöltjük az oldalt
        st.session_state["authenticated"] = False
        st.rerun()

    if st.sidebar.button("🚨 Káosz Kísérlet futtatása"):                # Káosz Kísérlet gomb a sidebar-on, amire kattintva szimulálunk egy hibás adatlekérést a get_data függvényben, és megjelenítjük a hibát a felhasználónak
        try:
            get_data(simulate_chaos=True)
        except Exception as e:
            st.error(f"Káosz Teszt Eredménye: {e}")
            st.info("A rendszer hibatűrő képessége igazolva: az applikáció stabil maradt.")

    # Adatok vizualizációja
    data = get_data()                                                   # Adatok lekérése a DynamoDB-ből a get_data függvény segítségével, amely visszaadja a készletadatokat egy listában
    if data:                                                            # Ha van adat, akkor egy DataFrame-be konvertáljuk a vizualizációhoz, és megjelenítjük egy oszlopdiagramon
        df = pd.DataFrame(data)
        df['quantity'] = df['quantity'].apply(float)
        
        # Táblázatos megjelenítés a készletlistáról
        display_df = df[['product_name', 'category', 'quantity']].copy()        # Csak a megjelenítéshez szükséges oszlopok kiválasztása egy új DataFrame-ben, hogy a táblázatban csak a termék neve, kategória és mennyiség jelenjen meg, a technikai PK/SK azonosítók ne
        display_df.columns = ['Név', 'Kategória', 'Darabszám']          # Oszlopnevek magyarra fordítása a táblázat megjelenítéséhez
        st.subheader("Aktuális készletlista (Táblázat)")                # Csak a lényeges oszlopokat mutatjuk, a PK/SK technikai azonosítókat nem
        st.dataframe(display_df, use_container_width=True, hide_index=True)  # Táblázat megjelenítése a termék neve, kategória és mennyiség oszlopokkal, a táblázat szélessége a rendelkezésre álló helyhez igazodik

        # Oszlopdiagram (Matplotlib integráció)
        st.subheader("Készletszintek termékenként")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(df['product_name'], df['quantity'], color='skyblue')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Adatkezelő űrlapok (CRUD műveletek a felületen)
        st.divider()                                                    # Vízszintes elválasztó a vizualizáció és az űrlapok között
        tab1, tab2 = st.tabs(["➕ Új termék felvétele", "✏️ Mennyiség módosítása"])      # Két fül létrehozása: az egyik az új termék hozzáadásához, a másik a meglévő termék mennyiségének módosításához
        
        with tab1:                                                      # Új termék hozzáadása űrlap
            with st.form("add_form"):                                   # Form elem létrehozása a Streamlit-ben, amely egy elkülönített űrlapot biztosít a termék hozzáadásához
                n = st.text_input("Név")                                # Szövegmező a termék nevének megadásához
                c = st.selectbox("Kategória", ["Elektronika", "Szerszámok", "Élelmiszerek", "Üdítők"])  # Legördülő lista a kategória kiválasztásához
                q = st.number_input("Mennyiség", min_value=0)           # Számmező a mennyiség megadásához, minimum érték 0
                if st.form_submit_button("Mentés"):                     # Mentés gomb, amire kattintva meghívjuk az add_new_product függvényt a megadott adatokkal, és megjelenítünk egy sikerüzenetet, majd újratöltjük az oldalt
                    add_new_product("BUD_01", n, q, c)

        with tab2:                                                      # Mennyiség módosítása űrlap
            prod_list = [item['product_name'] for item in data]         # Terméknevek listája a legördülő választóhoz, amely a lekért adatokból származik
            sel = st.selectbox("Termék kiválasztása", prod_list)        # Legördülő lista a termék kiválasztásához, amely a prod_list-ből származik
            item_data = next(i for i in data if i['product_name'] == sel)   # Kiválasztott termék adatainak lekérése a data listából, hogy megjeleníthessük a jelenlegi mennyiséget és azonosítókat a frissítéshez
            with st.form("update_form"):                                # Form elem létrehozása a Streamlit-ben, amely egy elkülönített űrlapot biztosít a mennyiség módosításához
                new_q = st.number_input("Új mennyiség", value=int(item_data['quantity']))   # Számmező az új mennyiség megadásához, alapértelmezett
                if st.form_submit_button("Frissítés"):                  # Frissítés gomb, amire kattintva meghívjuk az update_quantity függvényt a megadott adatokkal, és megjelenítünk egy sikerüzenetet, majd újratöltjük az oldalt
                    wh = item_data['PK'].split("#")[1]                  # A PK-ból kinyerjük a raktár azonosítóját a frissítéshez
                    pr = item_data['SK'].split("#")[1]                  # A SK-ból kinyerjük a termék azonosítóját a frissítéshez
                    update_quantity(wh, pr, new_q)                      # Mennyiség frissítése a DynamoDB-ben a megadott kulcsok alapján, és megjelenítünk egy sikerüzenetet, majd újratöltjük az oldalt
