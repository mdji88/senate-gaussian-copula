# Senate seat counts: independent races vs a Gaussian copula

**Question:** if each Senate race keeps the same win probability, how much does a shared "national mood" shock (dependence between races) change the odds of extreme outcomes?

This is the same structure as a credit portfolio: each race is a "name", a Democratic win is a "default", and the seat count is the portfolio loss. A one-factor Gaussian copula is the Vasicek / CDO-pricing model, applied here to elections.

> **Not a forecast.** `races.csv` contains *synthetic* inputs: a stylized map built from rating buckets (Safe / Likely / Lean / Toss-up) with assumed margins. It is tuned to be near a knife-edge so the effect of dependence is easy to see. Replace the margins with your own polling averages to run it on real data.

## Method

1. Each contested race *i* has an expected Democratic margin `mu_i` and a total uncertainty `SIGMA = 4.5` points, giving a marginal win probability `p_i = Phi(mu_i / SIGMA)`.
2. **Independent model (rho = 0):** each race is an independent coin flip with probability `p_i`.
3. **Gaussian copula:** `Z_i = sqrt(rho) * M + sqrt(1 - rho) * e_i`, with `M` the common national factor and `e_i` the race-specific noise. Race *i* is a Democratic win when `Phi(Z_i) < p_i`.
4. `rho` is the share of each race's variance driven by the national factor. Marginals are identical for every `rho`; only the dependence changes.
5. 400,000 simulations per `rho`. 34 Democratic-caucus seats are not up, and 51 seats are needed for control.

## Result (synthetic inputs)

| rho | mean seats | sd | 5th–95th pct | P(≥51) | P(≥55) | P(≤45) |
|---|---|---|---|---|---|---|
| 0.00 | 50.74 | 1.35 | 49–53 | 56.6% | 0.2% | 0.0% |
| 0.25 | 50.74 | 1.96 | 48–54 | 53.5% | 2.6% | 0.0% |
| 0.50 | 50.74 | 2.43 | 47–55 | 51.6% | 6.8% | 0.2% |
| 0.75 | 50.74 | 2.84 | 47–55 | 49.9% | 12.7% | 0.4% |

Running the script also writes `seat_distribution.svg`, a chart of the four seat-count distributions.

**Findings**

- The **expected seat count does not move** (50.74 in every row), because the marginals are fixed. Looking only at the average hides the effect of dependence.
- **Tails grow fast.** From rho = 0 to 0.75, the standard deviation of seats more than doubles, and the chance of a 55+ seat result rises from 0.2% to 12.7%. Independence badly understates blowout outcomes.
- **Near the majority line, dependence lowers the probability of the favored outcome:** P(≥51) falls from 56.6% to 49.9%, because probability mass moves from the middle into both tails. Correlation does not simply raise or lower the odds; it depends on which side of the threshold the mean sits.
- This mirrors tranche risk in credit: correlation barely changes expected loss but strongly changes the probability of extreme losses.

## Limitations

- `rho` is a sensitivity parameter, **not calibrated**. A natural next step is to estimate it from historical Senate polling errors across states.
- A Gaussian copula has no tail dependence. Real "wave" elections may cluster more than this model allows (compare Student-t copulas).
- State-level correlation is uniform here. In practice, similar states (demographics, region) move together more than dissimilar ones.
- Race margins and the total uncertainty are assumed, not estimated.

## Run it

```bash
pip install -r requirements.txt
python senate_copula.py
```

Outputs `results.csv` and `seat_distribution.svg` (the chart). Edit `races.csv` and the assumptions at the top of `senate_copula.py` to experiment.

## Next steps

- Calibrate `rho` from historical polling-error correlation across states.
- Swap in a Student-t copula and compare tail behavior.
- Add a state-similarity correlation matrix instead of a single factor.
