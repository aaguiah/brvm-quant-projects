"""
01_data_prep.py — Projet 03 : Portfolio Optimizer (brvm-quant-projects)
 
Rôle de ce module :
    1. Charger brvm_historique_ALL.csv (format long : Ticker, Date, OHLC, Volume)
    2. Calculer des statistiques par ticker (nb d'observations, volume médian, date de départ)
    3. Séparer l'univers en "inclus" / "exclu" selon deux seuils (historique, liquidité)
       -> les exclus ne sont PAS supprimés, ils sont tracés dans un fichier séparé
          pour une analyse de robustesse ultérieure (cf. décision méthodologique)
    4. Aligner les tickers inclus sur leurs dates communes (intersection stricte,
       pas de forward-fill — cf. discussion sur le biais de non-synchronicité)
    5. Calculer les log-rendements sur la base des prix alignés
    6. Sauvegarder : prix alignés, rendements, univers exclu, rapport de synthèse
 
Convention : les seuils sont centralisés en haut du fichier. Ils migreront vers
config.yaml une fois validés empiriquement sur plusieurs runs.
"""
 
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt
 
# ──────────────────────────────────────────────────────────────────────────
# CONFIGURATION — à terme dans config.yaml
# ──────────────────────────────────────────────────────────────────────────
 
DATA_PATH = Path("brvm_historique_ALL.csv")
OUTPUT_DIR = Path("outputs")  # créé automatiquement s.il n.existe pas
 
MAX_MISSING_PCT = 10.0      # % max de dates manquantes vs. calendrier complet du marché
                             # (capture à la fois IPO tardive et illiquidité chronique —
                             #  calibré empiriquement, cf. analyse tickers/dates communes)
MAX_WEIGHT = 0.10      # plafond de pondération maximum par titre
RF_ANNUAL = 0.06        # taux sans risque annuel (proxy obligations Etat UEMOA)
# ──────────────────────────────────────────────────────────────────────────
 
 
def load_data(path: Path) -> pd.DataFrame:
    """Charge le CSV brut et type correctement la colonne Date."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df
 
 
def compute_ticker_stats(df: pd.DataFrame) -> pd.DataFrame:
    calendrier_complet = df["Date"].nunique()
 
    stats = df.groupby("Ticker").agg(
        n_observations=("Date", "count"),
        date_debut=("Date", "min"),
        date_fin=("Date", "max"),
        volume_median=("Volume", "median"),          # descriptif, non filtrant
        pct_jours_volume_nul=("Volume", lambda x: (x == 0).mean() * 100),  # descriptif
    ).reset_index()
 
    stats["pct_missing_calendar"] = (1 - stats["n_observations"] / calendrier_complet) * 100
    return stats
 
 
def split_universe(stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare l'univers en inclus/exclus selon MAX_MISSING_PCT."""
    stats = stats.copy()
 
    exclu_mask = stats["pct_missing_calendar"] > MAX_MISSING_PCT
    stats["raison_exclusion"] = np.where(
        exclu_mask,
        stats["pct_missing_calendar"].round(1).astype(str) + "% de dates manquantes "
        f"(seuil: {MAX_MISSING_PCT}%)",
        ""
    )
 
    inclus = stats[~exclu_mask].drop(columns="raison_exclusion")
    exclus = stats[exclu_mask]
 
    return inclus, exclus
 
 
def align_dates(df: pd.DataFrame, tickers_inclus: list[str]) -> pd.DataFrame:
    sub = df[df["Ticker"].isin(tickers_inclus)]
    prix_large = sub.pivot(index="Date", columns="Ticker", values="Clôture")
 
    
    avant = len(prix_large)
    prix_large = prix_large.dropna(axis=0, how="any")
    apres = len(prix_large)
 
    print(f"Alignement temporel : {avant} dates -> {apres} dates communes "
          f"({avant - apres} dates supprimées, {(avant - apres) / avant * 100:.1f}%)")
 
    return prix_large
 
 
def compute_log_returns(prix_large: pd.DataFrame) -> pd.DataFrame:
    """Calcule les log-rendements à partir des prix alignés."""
    return np.log(prix_large / prix_large.shift(1)).dropna(how="all")
 
def compute_sample_stats(rendements: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Calcule le vecteur de rendements moyens et la matrice de covariance (estimateurs bruts)."""
    mu_hat = rendements.mean().values
    Sigma_hat = rendements.cov().values
    N = len(mu_hat)
    T = rendements.shape[0]
    return mu_hat, Sigma_hat, N, T

def min_variance_weights(Sigma_hat: np.ndarray) -> np.ndarray:
    """Calcule les poids du portefeuille à variance minimale (sans contrainte)."""
    N = Sigma_hat.shape[0]
    ones = np.ones(N)
    Sigma_inv = np.linalg.inv(Sigma_hat)
    w_mv = Sigma_inv @ ones / (ones @ Sigma_inv @ ones)
    return w_mv

def bayes_stein_shrinkage(rendements: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    """
    Calcule les rendements espérés shrinkés (Bayes-Stein, Jorion 1986).
    Retourne (mu_bs, phi, mu_mv) : les rendements shrinkés, l'intensité de
    shrinkage appliquée, et le rendement-cible (portefeuille min-variance).
    """
    mu_hat, Sigma_hat, N, T = compute_sample_stats(rendements)
    Sigma_inv = np.linalg.inv(Sigma_hat)

    w_mv = min_variance_weights(Sigma_hat)
    mu_mv = w_mv @ mu_hat                      # rendement du portefeuille cible

    ones = np.ones(N)
    ecart = mu_hat - mu_mv * ones
    lam = (N + 2) / (ecart @ Sigma_inv @ ecart)
    phi = lam / (T + lam)

    mu_bs = (1 - phi) * mu_hat + phi * mu_mv * ones
    return mu_bs, phi, mu_mv

def constant_correlation_target(Sigma_hat: np.ndarray) -> np.ndarray:
    """
    Construit la matrice cible du shrinkage : mêmes variances que l'échantillon,
    mais toutes les corrélations remplacées par leur moyenne (Ledoit-Wolf 2003).
    """
    N = Sigma_hat.shape[0]
    stds = np.sqrt(np.diag(Sigma_hat))
    corr = Sigma_hat / np.outer(stds, stds)

    off_diag = corr[~np.eye(N, dtype=bool)]
    r_bar = off_diag.mean()

    F = r_bar * np.outer(stds, stds)
    np.fill_diagonal(F, np.diag(Sigma_hat))
    return F

def shrinkage_intensity(rendements: pd.DataFrame, Sigma_hat: np.ndarray, F: np.ndarray) -> float:
    T = rendements.shape[0]
    X = rendements.values - rendements.values.mean(axis=0)
    outer_all = np.einsum('ti,tj->tij', X, X)
    pi_mat = ((outer_all - Sigma_hat) ** 2).mean(axis=0)
    pi_hat = pi_mat.sum()

    denom = ((F - Sigma_hat) ** 2).sum()
    return np.clip(pi_hat / (T * denom), 0, 1)

def shrink_covariance(rendements: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Calcule la matrice de covariance shrinkée (Ledoit-Wolf, cible constant correlation)."""
    Sigma_hat = rendements.cov().values
    F = constant_correlation_target(Sigma_hat)
    delta = shrinkage_intensity(rendements, Sigma_hat, F)
    Sigma_shrunk = delta * F + (1 - delta) * Sigma_hat
    return Sigma_shrunk, delta

def portfolio_variance(w: np.ndarray, Sigma: np.ndarray) -> float:
    """Variance du portefeuille : w'Σw — la même formule qu'à l'Étape 1 du Bloc Lagrange."""
    return w @ Sigma @ w

def negative_sharpe(w: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, rf: float) -> float:
    """Sharpe négatif — on minimise le négatif pour transformer 'maximiser' en 'minimiser'."""
    ret = w @ mu
    vol = np.sqrt(w @ Sigma @ w)
    return -(ret - rf) / vol

def optimize_min_variance(mu: np.ndarray, Sigma: np.ndarray, max_weight: float = MAX_WEIGHT) -> np.ndarray:
    N = len(mu)
    bounds = [(0, max_weight)] * N
    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
    w0 = np.ones(N) / N

    res = minimize(portfolio_variance, w0, args=(Sigma,), method='SLSQP',
                    bounds=bounds, constraints=constraints)
    return res.x

def optimize_max_sharpe(mu: np.ndarray, Sigma: np.ndarray, rf_annual: float = RF_ANNUAL,
                         max_weight: float = MAX_WEIGHT) -> np.ndarray:
    N = len(mu)
    rf_period = rf_annual / 252
    bounds = [(0, max_weight)] * N
    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
    w0 = np.ones(N) / N

    res = minimize(negative_sharpe, w0, args=(mu, Sigma, rf_period), method='SLSQP',
                    bounds=bounds, constraints=constraints)
    return res.x

def optimize_for_target_return(mu: np.ndarray, Sigma: np.ndarray, target: float,
                                 max_weight: float = MAX_WEIGHT) -> np.ndarray | None:
    """Portefeuille à variance minimale pour un rendement cible donné."""
    N = len(mu)
    bounds = [(0, max_weight)] * N
    constraints = [
        {'type': 'eq', 'fun': lambda w: w.sum() - 1},
        {'type': 'eq', 'fun': lambda w: w @ mu - target},
    ]
    w0 = np.ones(N) / N
    res = minimize(portfolio_variance, w0, args=(Sigma,), method='SLSQP',
                    bounds=bounds, constraints=constraints)
    return res.x if res.success else None
def generate_efficient_frontier(mu: np.ndarray, Sigma: np.ndarray, w_min_var: np.ndarray,
                                  max_weight: float = MAX_WEIGHT, n_points: int = 40):
    """
    Génère n_points le long de la frontière efficiente, entre le rendement du
    portefeuille min-variance et le rendement max réellement atteignable sous contrainte.
    """
    ret_min = w_min_var @ mu
# Rendement max atteignable : les meilleurs titres possibles, chacun au plafond
    n_min_assets = int(np.ceil(1 / max_weight))
    ret_max_feasible = np.sort(mu)[-n_min_assets:].sum() * max_weight
    ret_max = ret_min + (ret_max_feasible - ret_min) * 0.97   # marge de securite

    targets = np.linspace(ret_min, ret_max, n_points)
    frontier_vol, frontier_ret = [], []
    for t in targets:
        w = optimize_for_target_return(mu, Sigma, t, max_weight)
        if w is not None:
            frontier_vol.append(np.sqrt(w @ Sigma @ w) * np.sqrt(252) * 100)
            frontier_ret.append(t * 252 * 100)
    return frontier_vol, frontier_ret


def plot_efficient_frontier(frontier_vol, frontier_ret, w_min_var, w_max_sharpe,
                              mu: np.ndarray, Sigma: np.ndarray, rf_annual: float,
                              output_path: Path):
    vol_mv = np.sqrt(w_min_var @ Sigma @ w_min_var * 252) * 100
    ret_mv = (w_min_var @ mu) * 252 * 100
    vol_ms = np.sqrt(w_max_sharpe @ Sigma @ w_max_sharpe * 252) * 100
    ret_ms = (w_max_sharpe @ mu) * 252 * 100

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(frontier_vol, frontier_ret, 'b-', linewidth=2, label='Frontiere efficiente')
    ax.scatter([vol_mv], [ret_mv], color='green', s=100, zorder=5, label='Min-Variance')
    ax.scatter([vol_ms], [ret_ms], color='red', s=100, zorder=5, label='Max-Sharpe (tangent)')
    ax.plot([0, vol_ms], [rf_annual * 100, ret_ms], 'r--', linewidth=1, label='Capital Market Line')
    ax.set_xlabel('Volatilite annualisee (%)')
    ax.set_ylabel('Rendement annualise (%)')
    ax.set_title('Frontiere efficiente - BRVM (34 titres, Bayes-Stein + Ledoit-Wolf)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
 
    print("=" * 70)
    print("PROJET 03 — Data Prep")
    print("=" * 70)
 
    df = load_data(DATA_PATH)
    print(f"\nDonnées brutes chargées : {len(df)} lignes, {df['Ticker'].nunique()} tickers")
 
    stats = compute_ticker_stats(df)
    inclus_stats, exclus_stats = split_universe(stats)
 
    print(f"\nUnivers filtré : {len(inclus_stats)} tickers inclus, "
          f"{len(exclus_stats)} tickers exclus")
 
    if len(exclus_stats) > 0:
        print("\nTickers exclus (voir outputs/univers_exclu.csv pour le détail) :")
        for _, row in exclus_stats.sort_values("pct_missing_calendar", ascending=False).iterrows():
            print(f"  - {row['Ticker']}: {row['raison_exclusion']}")
 
    tickers_inclus = inclus_stats["Ticker"].tolist()
    prix_large = align_dates(df, tickers_inclus)
    rendements = compute_log_returns(prix_large)

     # ── Module 2 : Rendements espérés (Bayes-Stein) ────────────────────────
    mu_bs, phi, mu_mv = bayes_stein_shrinkage(rendements)
    mu_hat = rendements.mean().values

    print(f"\nShrinkage Bayes-Stein : phi={phi:.4f}, "
          f"cible (min-variance)={mu_mv*252*100:.2f}% annualise")
# ── Module 3 : Matrice de covariance (Ledoit-Wolf, cible constant correlation) ──
    Sigma_shrunk, delta = shrink_covariance(rendements)

    print(f"Shrinkage covariance : delta={delta:.4f}")
    pd.DataFrame({
        "ticker": rendements.columns,
        "mu_brut_annualise": mu_hat * 252 * 100,
        "mu_bayes_stein_annualise": mu_bs * 252 * 100,
    }).to_csv(OUTPUT_DIR / "rendements_esperes.csv", index=False)

# ── Module 4 : Optimisation (min-variance et max-Sharpe) ────────────────
    w_min_var = optimize_min_variance(mu_bs, Sigma_shrunk)
    w_max_sharpe = optimize_max_sharpe(mu_bs, Sigma_shrunk)

    ret_mv = w_min_var @ mu_bs * 252
    vol_mv = np.sqrt(w_min_var @ Sigma_shrunk @ w_min_var * 252)
    ret_ms = w_max_sharpe @ mu_bs * 252
    vol_ms = np.sqrt(w_max_sharpe @ Sigma_shrunk @ w_max_sharpe * 252)
    sharpe_ms = (ret_ms - RF_ANNUAL) / vol_ms

    print(f"\nPortefeuille Min-Variance : rendement={ret_mv*100:.2f}%, "
          f"volatilite={vol_mv*100:.2f}%")
    print(f"Portefeuille Max-Sharpe   : rendement={ret_ms*100:.2f}%, "
          f"volatilite={vol_ms*100:.2f}%, Sharpe={sharpe_ms:.2f}")
    
 # ── Module 5 : Frontière efficiente ──────────────────────────────────────
    frontier_vol, frontier_ret = generate_efficient_frontier(mu_bs, Sigma_shrunk, w_min_var)
    plot_efficient_frontier(frontier_vol, frontier_ret, w_min_var, w_max_sharpe,
                             mu_bs, Sigma_shrunk, RF_ANNUAL,
                             OUTPUT_DIR / "efficient_frontier.png")

    print(f"\nFrontiere efficiente : {len(frontier_vol)} points generes, "
          f"sauvegardee dans {OUTPUT_DIR}/efficient_frontier.png")
 
 
    # ── Sauvegardes ──────────────────────────────────────────────────────
    prix_large.to_csv(OUTPUT_DIR / "prix_alignes.csv")
    rendements.to_csv(OUTPUT_DIR / "rendements_log.csv")
    exclus_stats.to_csv(OUTPUT_DIR / "univers_exclu.csv", index=False)
    inclus_stats.to_csv(OUTPUT_DIR / "univers_inclus_stats.csv", index=False)
    pd.DataFrame(
        Sigma_shrunk, index=rendements.columns, columns=rendements.columns
    ).to_csv(OUTPUT_DIR / "covariance_shrunk.csv")

    pd.DataFrame({
        "ticker": rendements.columns,
        "poids_min_variance": w_min_var,
        "poids_max_sharpe": w_max_sharpe,
    }).to_csv(OUTPUT_DIR / "poids_optimaux.csv", index=False)
 
    print(f"\nFichiers sauvegardés dans {OUTPUT_DIR}/ :")
    print("  - prix_alignes.csv        (prix de clôture, dates communes)")
    print("  - rendements_log.csv      (log-rendements, base de travail pour 02 et 03)")
    print("  - univers_exclu.csv       (tickers exclus + raison, pour robustesse)")
    print("  - univers_inclus_stats.csv (stats descriptives de l'univers retenu)")
 
    print(f"\nUnivers final pour l'optimisation : {rendements.shape[1]} tickers, "
          f"{rendements.shape[0]} observations de rendements")
    print("  - rendements_esperes.csv (rendements bruts vs. shrinkés Bayes-Stein)")
    print("  - covariance_shrunk.csv  (matrice de covariance shrinkee, base pour l'optimiseur)")
    print("  - poids_optimaux.csv    (poids min-variance et max-Sharpe)")
 
 
if __name__ == "__main__":
    main()