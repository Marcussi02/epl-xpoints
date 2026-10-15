# epl-xpoints

Premier League match predictions, scored honestly against the future, feeding a Fantasy
Premier League optimiser.

- **Predict:** every match as home/draw/away probabilities and a full scoreline
  distribution, using Elo and Dixon–Coles models written from scratch (maximum likelihood
  with an analytic gradient).
- **Prove it:** walk-forward backtests that refit every week on the past only. A leakage
  guard raises an error on any overlap, scoring uses proper rules (RPS, log loss, Brier,
  calibration), and a paired bootstrap says whether one model really beats another.
- **Use it:** expected FPL points per player per gameweek, fed to an exact
  integer-programming planner for squad, XI, captain and multi-week transfers. Whole
  seasons are replayed to see what the decisions would have scored.

## Headline results

All results are on held-out seasons that no setting was tuned on. Full method and caveats are in **[REPORT.md](REPORT.md)**.

**Match prediction** (2022-23 to 2025-26, 1,520 matches):

| Model | RPS ↓ | Log loss ↓ | Accuracy |
|---|---|---|---|
| Historical frequencies (no skill) | 0.2318 | 1.069 | 44.5% |
| Elo + ordered logit | 0.2057 | 0.995 | 52.1% |
| **Dixon–Coles** | **0.2045** | **0.990** | 52.0% |

- **Both models clearly beat no skill.** Elo vs frequency: −0.026 RPS, 95% interval
  [−0.034, −0.019].
- **Dixon–Coles and Elo are statistically tied.**

**FPL expected points** (2024-25, 25,159 player-gameweeks), compared with FPL's own
published forecast:

| Forecast | MAE ↓ | Rank correlation, players who played ↑ | Top 50 picks: actual points ↑ |
|---|---|---|---|
| **This model** | **1.023** | **0.351** | **4.30** |
| FPL's published forecast | 1.142 | 0.251 | 3.90 |

Along the way I found that the community dataset's `xP` column is recorded a gameweek
late, which makes FPL's forecast look far better than it is. [REPORT.md](REPORT.md)
shows the test and the fix.

**Season replays** (same planner for every forecast):
- The model beats recent form in both seasons.
- Against FPL's published forecast it splits: +105 points in 2024-25, −10 in 2023-24.
- One season of FPL points is too noisy to separate them.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

epl backtest            # match models, walk-forward, held-out seasons
epl tune                # how the settings were chosen (validation season only)
epl xp                  # FPL expected points vs FPL's own forecast
epl season              # replay the 2024-25 FPL season with each forecast
pytest -q               # 56 tests, including property tests for the optimiser
```

Data downloads on first use and is cached in `data/cache/`.

## How it works

```
epl/
  data/     downloads with an on-disk cache, tidy match and player tables, canonical team names
  eval/     scoring rules, calibration, walk-forward backtest, tuning, paired bootstrap
  models/   frequency and market baselines, Elo, Dixon–Coles
  fpl/      expected points, integer-programming planner, season replay
  cli.py    the epl command
```

- **Dixon–Coles:** Poisson goals per team with attack and defence strengths, home
  advantage, the low-score correction ρ, exponential time decay and light ridge
  shrinkage. Fitted by L-BFGS-B with an analytic gradient, which a test checks against
  numerical differentiation. Another test recovers known parameters from simulated
  seasons.
- **Elo:** margin-of-victory updates and a promoted-club prior, with an ordered-logit
  link fitted to turn rating gaps into home/draw/away probabilities.
- **FPL planner:** 0/1 variables for squad, XI, captain, buy and sell over a rolling
  horizon. The constraints are squad quotas, the 3-per-club limit, budget and selling
  prices, valid formations and free-transfer roll-over, solved exactly by HiGHS through
  `scipy.optimize.milp`.

## Data and credits

- **Match results and FPL player history:** the
  [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)
  dataset (MIT licence).
- **Bookmaker odds and older results:** [football-data.co.uk](https://www.football-data.co.uk),
  free for non-commercial use. This adds the market benchmark.
- **Live players, prices and fixtures:** the official FPL API. It is public but
  unofficial, so requests are cached and rate-limited.

This is an independent project, not affiliated with the Premier League or FPL.

## Status

- [x] Data layer
- [x] Evaluation harness
- [x] Elo and Dixon–Coles models
- [x] FPL expected points, optimiser and season replays
- [ ] Market benchmark from bookmaker odds, and multi-season validation on the longer history
- [ ] Weekly live predictions and FPL picks, published as a static site

## Licence

MIT
