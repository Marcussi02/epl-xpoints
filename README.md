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

**Match prediction** (2022-23 to 2025-26, 1,520 matches; settings chosen on 2016-17 to 2021-22):

| Model | RPS ↓ | Log loss ↓ | Accuracy |
|---|---|---|---|
| Historical frequencies (no skill) | 0.2316 | 1.068 | 44.5% |
| Elo + ordered logit | 0.2039 | 0.988 | 52.4% |
| **Dixon–Coles** | **0.2030** | **0.986** | 52.6% |
| Bookmaker closing odds | 0.1947 | 0.960 | 55.1% |

- **Both models clearly beat no skill.** Dixon–Coles vs frequency: −0.029 RPS, 95%
  interval [−0.035, −0.022].
- **From scorelines alone, Dixon–Coles closes 78% of the gap to the closing odds** and
  is as well calibrated. The market's remaining edge (+0.008 RPS, [+0.005, +0.011]) is
  what team news and lineups are worth.
- **Dixon–Coles and Elo are statistically tied.**

**FPL expected points** (2024-25, 25,159 player-gameweeks), compared with FPL's own
published forecast:

| Forecast | MAE ↓ | Rank correlation, players who played ↑ | Top 50 picks: actual points ↑ |
|---|---|---|---|
| **This model** | **1.012** | **0.366** | **4.40** |
| FPL's published forecast | 1.142 | 0.251 | 3.90 |

Along the way I found two data traps, documented in [REPORT.md](REPORT.md):
- **Reused player ids:** FPL renumbers players every season, so joining seasons by id
  mixes up different people. Rows are now keyed by FPL's stable player code.
- **A leaky benchmark:** the community dataset's `xP` column is recorded a gameweek late,
  which makes FPL's forecast look far better than it is.

**Season replay**, 2024-25 (same planner for every forecast):

| Forecast | Season points |
|---|---|
| **This model** | **2,102** |
| Recent form | 1,928 |
| FPL's published forecast | 1,744 |

- **Against FPL's forecast:** +358 points, 95% interval [+151, +550].
- **Against recent form:** +174 points, 95% interval [−29, +370], so not quite
  significant over one season.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

epl backtest            # match models, walk-forward, held-out seasons
epl tune                # how the settings were chosen (validation seasons only)
epl xp                  # FPL expected points vs FPL's own forecast
epl season              # replay the 2024-25 FPL season with each forecast
epl site                # this gameweek's predictions and picks, as site/index.html
pytest -q               # 65 tests, including property tests for the optimiser
```

Data downloads on first use and is cached in `data/cache/`.

## How it works

```
epl/
  data/     downloads with an on-disk cache, tidy match and player tables, canonical team names
  eval/     scoring rules, calibration, walk-forward backtest, tuning, paired bootstrap
  models/   frequency and market baselines, Elo, Dixon–Coles
  fpl/      expected points, integer-programming planner, season replay
  report/   the static page: one self-contained HTML file, inline SVG, light and dark
  live.py   this season from the FPL API, with injury flags, for the upcoming gameweek
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

- **Match results and bookmaker odds, 2000-01 onwards:**
  [football-data.co.uk](https://www.football-data.co.uk), free for non-commercial use.
- **FPL player history:** the
  [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)
  dataset (MIT licence).
- **Live players, prices and fixtures:** the official FPL API. It is public but
  unofficial, so requests are cached and rate-limited.

This is an independent project, not affiliated with the Premier League or FPL.

## Status

- [x] Data layer
- [x] Evaluation harness
- [x] Elo and Dixon–Coles models
- [x] FPL expected points, optimiser and season replays
- [x] Market benchmark from bookmaker closing odds, and six-season validation on 26 seasons of history
- [x] Live predictions and FPL picks as a static page, rebuilt daily by GitHub Actions

## Licence

MIT
