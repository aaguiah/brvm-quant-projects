# PROJET 01 — SCRAPER BRVM
# Auteur : Anges Aguiah
# Date : Mai 2026
# ============================================

import pandas as pd
import numpy as np
import requests
import urllib3
import time
from bs4 import BeautifulSoup
from datetime import datetime

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

print("✅ Librairies importées avec succès")
print(f"Date d'exécution : {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# ============================================
# LISTE DES TICKERS BRVM
# ============================================

tickers = [
    "SDSC", "BOAB", "BOAS", "BOABF", "BOAN",
    "BOAC", "BOAM", "BICB", "ECOC", "CBIBF",
    "BNBC", "TTLC", "TTLS", "UNLC", "UNXC",
    "SHEC", "ETIT", "PRSC", "SNTS", "SCRC",
    "SLBC", "SIBC", "SOGC", "SDCC", "SMBC",
    "STBC", "SICC", "CABC", "SGBC", "STAC",
    "ABJC", "SPHC", "PALC", "SAFC", "ORAC",
    "ORGT", "ONTBF", "NSBC", "NTLC", "NEIC",
    "LNBB", "FTSC", "SIVC", "SEMC", "CIEC",
    "CFAC", "BICC"
]

extensions = {
    "ONTBF": "bf", "CBIBF": "bf", "BOAB": "bj",
    "LNBB": "bj", "BICB": "bj", "BOABF": "bf",
    "BOAS": "sn", "SNTS": "sn", "TTLS": "sn",
    "BOAN": "ne",
    "BOAM": "ml",
    "ETIT": "tg", "ORGT": "tg",
    "SOGC": "ci"
}

print(f"📋 {len(tickers)} valeurs BRVM identifiées")

# ============================================
# FONCTION — Scraper l'historique d'une valeur
# ============================================

def scraper_historique(ticker):
    ext = extensions.get(ticker, "ci")
    url = f"https://www.sikafinance.com/marches/historiques/{ticker}.{ext}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        reponse = requests.get(url, headers=headers, timeout=10, verify=False)
        if reponse.status_code != 200:
            print(f"❌ {ticker} : statut {reponse.status_code}")
            return None

        soup = BeautifulSoup(reponse.content, "html.parser")
        tableau = soup.find("table")

        if tableau is None:
            print(f"❌ {ticker} : aucun tableau trouvé")
            return None

        lignes = tableau.find_all("tr")
        donnees = []
        for ligne in lignes:
            cellules = ligne.find_all(["td", "th"])
            donnees.append([c.get_text(strip=True) for c in cellules])

        df = pd.DataFrame(donnees[1:], columns=donnees[0])
        df["Ticker"] = ticker
        print(f"✅ {ticker} : {df.shape[0]} lignes extraites")
        return df

    except Exception as e:
        print(f"❌ {ticker} : erreur — {e}")
        return None

# ============================================
# BOUCLE — Scraper toutes les valeurs BRVM
# ============================================

tous_les_df = []

for ticker in tickers:
    df = scraper_historique(ticker)
    if df is not None:
        tous_les_df.append(df)
    time.sleep(1)

df_brvm_complet = pd.concat(tous_les_df, ignore_index=True)

print(f"\n🎯 BASE DE DONNÉES COMPLÈTE")
print(f"Total lignes : {df_brvm_complet.shape[0]}")
print(f"Total valeurs : {df_brvm_complet['Ticker'].nunique()}")

df_brvm_complet.to_csv("brvm_historique.csv", index=False, encoding="utf-8-sig")
print("💾 Fichier sauvegardé : brvm_historique.csv")
# ============================================
# RÉCUPÉRATION DES VALEURS MANQUANTES
# ============================================

manquants = ["ORAC", "CBIBF"]

for ticker in manquants:
    print(f"🔄 Nouvelle tentative : {ticker}")
    time.sleep(3)  # pause plus longue
    df = scraper_historique(ticker)
    if df is not None:
        df_brvm_complet = pd.concat([df_brvm_complet, df], ignore_index=True)

print(f"\n🎯 TOTAL FINAL")
print(f"Total lignes : {df_brvm_complet.shape[0]}")
print(f"Total valeurs : {df_brvm_complet['Ticker'].nunique()}")

df_brvm_complet.to_csv("brvm_historique.csv", index=False, encoding="utf-8-sig")
print("💾 Fichier mis à jour : brvm_historique.csv")