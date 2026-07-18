# BRVM Quant Projects
**Auteur : Anges Aguiah** | Portfolio Manager | CFA Candidate

Projets quantitatifs appliqués aux marchés financiers de la BRVM et de la zone UEMOA.

## Projet 01 — BRVM Data Scraper
Pipeline automatisé de collecte de données historiques sur les 47 valeurs cotées à la BRVM (8 pays UEMOA).

**Technologies :** Python, BeautifulSoup, Pandas, Requests

**Output :** 3135 lignes de données historiques (Date, Clôture, Volume, Variation)

**Source :** SikaFinance.com

## Projets à venir
- 04 — VaR Model
- 05 — Factor Investing Model (Fama-French adapté BRVM)

## Projet 02 — Backtesting Engine Mean-Reversion Z-score

**Stratégie :** Mean-Reversion basée sur le Z-score (fenêtre 20 jours)

**Technologies :** Python, Pandas, NumPy, Matplotlib

**Résultats :** FTSC +14.3%, NTLC +10.8%, CABC +9.8% sur la période

**Données :** 3 mois d'historique (limitation source SikaFinance)

**Note :** En environnement professionnel, l'historique serait étendu à 3-5 ans via flux Bloomberg ou données officielles BRVM pour une significativité statistique robuste.

## Projet 03 — Portfolio Optimizer (Moyenne-Variance)

Optimiseur de portefeuille mean-variance appliqué à 34 valeurs BRVM, avec correction du risque d'estimation via shrinkage bayésien (rendements) et Ledoit-Wolf (covariance).

**Univers** : 34 titres retenus sur 47 (seuil de couverture calendaire à 90%, exclusion des IPO récentes et titres illiquides — cf. `univers_exclu.csv` pour le détail et la justification par ticker)

**Méthodologie** :
- Rendements espérés : shrinkage Bayes-Stein (Jorion, 1986), cible = rendement du portefeuille à variance minimale
- Matrice de covariance : shrinkage Ledoit-Wolf, cible constant-correlation (Ledoit-Wolf, 2003), formule d'intensité simplifiée
- Contraintes : long-only, plafond de 10% par titre
- Résolution : programmation quadratique sous contraintes (`scipy.optimize`, SLSQP)

**Technologies** : Python, Pandas, NumPy, SciPy, Matplotlib

**Résultats** :
- Portefeuille Min-Variance : rendement 25,56% / volatilité 12,57% (annualisés)
- Portefeuille Max-Sharpe (tangent) : rendement 28,10% / volatilité 11,29%, Sharpe 1,96

**Limites** : le Sharpe élevé (1,96) reflète probablement un biais de non-synchronicité propre aux marchés frontières — l'illiquidité résiduelle de certains titres tend à surestimer les rendements mesurés et sous-estimer la volatilité mesurée, dans le même sens. À interpréter comme un signal d'optimisation relatif, pas comme une performance espérée réaliste sans ajustement supplémentaire.
