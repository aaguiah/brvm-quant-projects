# ============================================
# PROJET 02 — BACKTESTING ENGINE BRVM
# Stratégie : Mean-Reversion Z-score
# Auteur : Anges Aguiah
# Date : Juin 2026
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

print("Librairies importées")

# CHARGEMENT DES DONNÉES

df = pd.read_csv("brvm_historique_ALL.csv", sep=",")

print(f"Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(df.head())

# NETTOYAGE DES DONNÉES

df["Date"] = pd.to_datetime(df["Date"])

df["Clôture"] = pd.to_numeric(df["Clôture"], errors="coerce")

df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

print(f"Types des colonnes :")
print(df.dtypes)

# CALCUL DU Z-SCORE

fenetre = 20  # fenêtre de calcul du z-score
df["moyenne"] = df.groupby("Ticker")["Clôture"].transform(lambda x: x.rolling(fenetre).mean())
df["ecart_type"] = df.groupby("Ticker")["Clôture"].transform(lambda x: x.rolling(fenetre).std())
df["zscore"] = (df["Clôture"] - df["moyenne"]) / df["ecart_type"]

print(df[["Date", "Ticker", "Clôture", "moyenne", "zscore"]].dropna().head(10))

# SIGNAUX DE TRADING

df["signal"] = 0
df.loc[df["zscore"] < -2, "signal"] = 1
df.loc[df["zscore"] > 2, "signal"] = -1

print(f"Signaux générés :")
print(df["signal"].value_counts())

# CALCUL DES RENDEMENTS

df["rendement"] = df.groupby("Ticker")["Clôture"].transform(
    lambda x: x.pct_change()
)

df["rendement_strategie"] = df["signal"].shift(1) * df["rendement"]

print(df[["Date", "Ticker", "Clôture", "signal", "rendement", "rendement_strategie"]].dropna().head(10))

# PERFORMANCE DE LA STRATEGIE

performance = df.groupby("Ticker")["rendement_strategie"].sum().reset_index()
performance.columns = ["Ticker", "rendement_total"]
performance = performance.sort_values("rendement_total", ascending=False)

print("Top 10 valeurs — Strategie Mean-Reversion :")
print(performance.head(10))
print(f"\nRendement moyen toutes valeurs : {performance['rendement_total'].mean():.2%}")

# VISUALISATION

ticker_test = "SIBC"
df_ticker = df[df["Ticker"] == ticker_test].dropna(subset=["zscore"])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(df_ticker["Date"], df_ticker["Clôture"], label="Prix cloture")
ax1.set_title(f"{ticker_test} — Prix de cloture")
ax1.set_ylabel("Prix (FCFA)")
ax1.legend()

ax2.plot(df_ticker["Date"], df_ticker["zscore"], label="Z-score", color="orange")
ax2.axhline(y=2, color="red", linestyle="--", label="Seuil vente (+2)")
ax2.axhline(y=-2, color="green", linestyle="--", label="Seuil achat (-2)")
ax2.axhline(y=0, color="gray", linestyle="-")
ax2.set_title(f"{ticker_test} — Z-score")
ax2.set_ylabel("Z-score")
ax2.legend()

plt.tight_layout()
plt.savefig("backtest_zscore_SIBC.png")
plt.show()
print("Graphique sauvegarde")

