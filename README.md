# BRVM Quant Projects
**Auteur : Anges Aguiah** | Portfolio Manager | CFA Candidate

Projets quantitatifs appliqués aux marchés financiers de la BRVM et de la zone UEMOA.

## Projet 01 — BRVM Data Scraper
Pipeline automatisé de collecte de données historiques sur les 47 valeurs cotées à la BRVM (8 pays UEMOA).

**Technologies :** Python, BeautifulSoup, Pandas, Requests

**Output :** 3135 lignes de données historiques (Date, Clôture, Volume, Variation)

**Source :** SikaFinance.com

## Projets à venir
- 02 — Backtesting Engine
- 03 — Portfolio Optimizer (Mean-Variance)
- 04 — VaR Model
- 05 — Factor Investing Model (Fama-French adapté BRVM)

## Projet 02 — Backtesting Engine Mean-Reversion Z-score

**Stratégie :** Mean-Reversion basée sur le Z-score (fenêtre 20 jours)

**Technologies :** Python, Pandas, NumPy, Matplotlib

**Résultats :** FTSC +14.3%, NTLC +10.8%, CABC +9.8% sur la période

**Données :** 3 mois d'historique (limitation source SikaFinance)

**Note :** En environnement professionnel, l'historique serait étendu à 3-5 ans via flux Bloomberg ou données officielles BRVM pour une significativité statistique robuste.