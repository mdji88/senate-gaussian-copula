"""Senate seat-count distribution: independent races vs a one-factor Gaussian copula.

Question: if every race has the same marginal win probability, how much does
*dependence* (a shared "national mood" shock) change the chance of extreme
outcomes and of crossing the majority line?

Run from this folder:  python senate_copula.py
Reads data/races.csv. Outputs: results/results.csv, charts/seat_distribution.svg
and a printed summary.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["svg.fonttype"] = "none"

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"; OUT = HERE / "results"; FIG = HERE / "charts"
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

# ---- assumptions (edit these) ------------------------------------------------
SIGMA_TOTAL = 4.5        # total sd of a race's margin around its mean (points)
HOLDOVER_D = 34          # Democratic-caucus seats NOT up this cycle
MAJORITY = 51            # seats needed for control if the VP is of the other party
RHOS = [0.0, 0.25, 0.5, 0.75]   # share of variance driven by the national factor
N_SIMS = 400_000
SEED = 7
# ------------------------------------------------------------------------------

races = pd.read_csv(DATA / "races.csv")
mu = races["dem_margin_mean_pts"].to_numpy()
p = norm.cdf(mu / SIGMA_TOTAL)          # marginal D win probability per race
n = len(p)
rng = np.random.default_rng(SEED)


def simulate(rho: float) -> np.ndarray:
    """D seats won among contested races under a one-factor Gaussian copula.

    Z_i = sqrt(rho) * M + sqrt(1 - rho) * e_i  with M, e_i ~ N(0, 1).
    Race i is a D win when U_i = Phi(Z_i) < p_i, so each marginal win
    probability stays exactly p_i and only the dependence changes.
    rho = 0 is the independent-races model.
    """
    m = rng.standard_normal((N_SIMS, 1))
    e = rng.standard_normal((N_SIMS, n))
    z = np.sqrt(rho) * m + np.sqrt(1 - rho) * e
    return (norm.cdf(z) < p).sum(axis=1)


rows, dists = [], {}
for rho in RHOS:
    seats = HOLDOVER_D + simulate(rho)
    dists[rho] = seats
    rows.append({
        "rho": rho,
        "mean_seats": seats.mean(),
        "sd_seats": seats.std(),
        "p05": np.percentile(seats, 5),
        "p95": np.percentile(seats, 95),
        "P(D>=51)": (seats >= MAJORITY).mean(),
        "P(D>=55)": (seats >= 55).mean(),
        "P(D<=45)": (seats <= 45).mean(),
    })

out = pd.DataFrame(rows).round(4)
out.to_csv(OUT / "results.csv", index=False)
print(f"Sum of marginal win probabilities: {p.sum():.3f} "
      f"(expected D seats = {HOLDOVER_D + p.sum():.2f} for every rho)\n")
print(out.to_string(index=False))

# ---- chart -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.8))
colors = {0.0: "#9aa5b4", 0.25: "#4fb3a9", 0.5: "#e0a030", 0.75: "#c0504d"}
bins = np.arange(HOLDOVER_D - 0.5, HOLDOVER_D + n + 1.5)
for rho in RHOS:
    counts, _ = np.histogram(dists[rho], bins=bins)
    ax.plot(np.arange(HOLDOVER_D, HOLDOVER_D + n + 1), counts / N_SIMS,
            label=f"rho = {rho}", color=colors[rho], lw=2)
ax.axvline(MAJORITY - 0.5, color="black", ls="--", lw=1)
ax.text(MAJORITY - 0.3, ax.get_ylim()[1] * 0.92, "51-seat majority", fontsize=9)
ax.set_xlim(40, 62)
ax.set_xlabel("Democratic-caucus seats after the election (illustrative inputs)")
ax.set_ylabel("Probability")
ax.set_title("Same marginals, different dependence: the seat distribution fattens as rho rises")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "seat_distribution.svg")
print("\nSaved results/results.csv and charts/seat_distribution.svg")
