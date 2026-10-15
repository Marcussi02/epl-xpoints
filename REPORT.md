# Results

Two questions, answered without letting any model see the future:

1. **Match prediction:** how well can a model built only from past scorelines predict
   Premier League matches?
2. **Fantasy Premier League:** do those predictions make better FPL decisions?

Reproduce everything here with:

```bash
epl tune       # match models: validation season 2021-22 only
epl backtest   # match models: test seasons 2022-23 to 2025-26
epl xp         # FPL expected points vs FPL's own forecast, 2024-25
epl season     # FPL season replay, 2024-25
```

# Part 1: match prediction

## Method

- **Data:** every Premier League result from 2019-20 to 2025-26 (2,660 matches), from the
  MIT-licensed FPL community dataset.
- **Walk-forward:** for each week of a test season, every model is refitted on matches that
  finished before that week, then predicts the week's fixtures from pre-match information
  only. The harness raises an error if any training match overlaps the week being predicted.
- **Split:**
  - 2019-20 and 2020-21: history only.
  - 2021-22: validation; hyper-parameters are chosen here, by mean RPS.
  - 2022-23 to 2025-26: test; 1,520 matches, looked at once with frozen settings.
- **Metrics:**
  - Ranked probability score (RPS, the standard for ordered football outcomes).
  - Log loss and Brier score.
  - Accuracy (for intuition only).
  - Expected calibration error.
- **Significance:** the paired bootstrap resamples whole weeks of matches, so a gap counts
  as real only when its 95% interval excludes zero.

## Held-out results (2022-23 to 2025-26)

| Model | RPS ↓ | Log loss ↓ | Brier ↓ | Accuracy | Calibration error ↓ |
|---|---|---|---|---|---|
| Frequency (no skill) | 0.2318 | 1.069 | 0.647 | 44.5% | 0.013 |
| Elo + ordered logit | 0.2057 | 0.995 | 0.594 | 52.1% | 0.024 |
| **Dixon–Coles** | **0.2045** | **0.990** | **0.590** | 52.0% | **0.013** |

RPS by season:

| Season | Frequency | Elo | Dixon–Coles |
|---|---|---|---|
| 2022-23 | 0.2304 | **0.2113** | 0.2141 |
| 2023-24 | 0.2341 | **0.1890** | 0.1897 |
| 2024-25 | 0.2348 | 0.2079 | **0.2045** |
| 2025-26 | 0.2279 | 0.2146 | **0.2097** |

Paired differences in RPS (negative = first model better):

| Comparison | Mean difference | 95% interval |
|---|---|---|
| Elo − Frequency | −0.0261 | −0.0335 to −0.0185 |
| Dixon–Coles − Frequency | −0.0273 | −0.0340 to −0.0206 |
| Dixon–Coles − Elo | −0.0012 | −0.0039 to +0.0015 |

## What the numbers say

- **Both models have real skill.** Each cuts RPS by about 11–12% against
  frequency-only guessing, and the interval is nowhere near zero.
- **Dixon–Coles and Elo are statistically tied.** Dixon–Coles is a little better on
  average and better calibrated, but four seasons cannot separate them. Dixon–Coles is
  still the one to build on: it predicts whole scorelines, which the FPL optimiser needs
  for clean-sheet and goal expectations.
- **Elo's home-advantage setting barely matters.** The ordered-logit link learns home
  advantage from data, absorbing it into its thresholds.
- **Tuning on a single season overfit Elo.** Its tuned settings (K = 10, no season
  regression) did worse than the untuned defaults (K = 20, regression 0.2) on three of
  the four test seasons, and its calibration error doubled. I report the tuned numbers
  anyway: re-choosing settings after seeing test results would leak the test set. The
  fix is to validate on several seasons, which needs the longer football-data.co.uk
  history (back to 2000).
- **2020-21 is an outlier worth knowing about.** Matches were played without crowds, and
  it is the only season where away wins (40.3%) outnumbered home wins (37.9%). It sits in
  the training history, and its weight fades with the time decay.

# Part 2: Fantasy Premier League

## Expected points per player

- **Inputs from the match model:** Dixon–Coles supplies each fixture's expected goals for
  and against and the clean-sheet probability.
- **Inputs from each player's earlier gameweeks:**
  - Minutes: the probability of playing, and of playing 60+.
  - Share of the team's xG and xA while on the pitch, shrunk towards his position's
    typical share.
  - A per-90 residual covering bonus, saves, cards and defensive contributions.
- **Settings:** one tuning grid on 2023-24, which only shortened the minutes window from
  6 gameweeks to 4. 2024-25 is held out.

**Two data traps, found and fixed:**

- **Player ids are reused every season.** FPL renumbers players each summer: element 384
  was a Liverpool outfielder in 2025-26 and Man City's goalkeeper in 2026-27. Joining
  last season's rows to this season's players by id silently attached *every* one of
  them to the wrong person (100% in both seasons checked). Rows now carry FPL's stable
  player `code`, and history is re-keyed by code before it is combined. All numbers
  below are after the fix.
- **A leaky benchmark.** The community dataset records FPL's own expected points (`xP`)
  for every player and gameweek, and at first it looked unbeatable: a Spearman
  correlation of 0.58 with the gameweek's points among players who played. Consecutive
  gameweeks' points correlate only 0.21, though, and the same `xP` correlates just 0.22
  with the *next* gameweek's points. So the value on a gameweek's row was recorded
  after that gameweek: it is FPL's forecast for the following one, built from form that
  already includes the row's result. Shifted to the gameweek it was published for, it
  becomes a fair benchmark.

Compared on the same player-gameweeks:

| Season | Forecast | MAE ↓ | Rank correlation ↑ | Rank correlation, players who played ↑ | Top 50 picks per gameweek: actual points ↑ |
|---|---|---|---|---|---|
| **2024-25 (test)** | **This model** | **1.012** | **0.714** | **0.366** | **4.40** |
| | FPL's published forecast | 1.142 | 0.581 | 0.251 | 3.90 |
| 2023-24 (validation) | This model | 0.981 | 0.711 | 0.371 | 4.67 |
| | FPL's published forecast | 1.122 | 0.593 | 0.265 | 3.89 |

FPL's forecast also knows injury and suspension news, which the historical files don't
record. Live, the planner uses FPL's injury flags, so this model gains that
information too.

## Squad selection and transfers

- **The optimiser:** an exact integer program (HiGHS through `scipy.optimize.milp`).
  - Picks a 15-player squad (2 GK, 5 DEF, 5 MID, 3 FWD, at most 3 per club,
    £100m), a valid XI and a captain.
  - Plans transfers over a rolling horizon, with free transfers rolling over up to five.
  - Property tests generate random player pools and check that every plan obeys every rule.
- **The replay:** gameweeks 2–38 of a past season, with each week's decisions made at
  the deadline. Real points are counted with captaincy, auto-substitutions, hits and
  FPL selling prices. Chips are not used.
- **Tuning on 2023-24:** planning horizon (1 or 3 gameweeks) and how much the planner
  charges itself per hit (4, 6, 8 or 12). The best was the simplest: a 1-gameweek
  horizon at the real 4-point cost (2,200 points). Results across the grid ranged from
  2,091 to 2,200, so the choice matters less than the forecasts do. (With the id bug
  still in the data, a penalty of 8 had looked best; that finding did not survive the
  fix.)

Same frozen planner for every forecast:

| Season | Forecast | Season points | Transfers | Hits |
|---|---|---|---|---|
| **2024-25 (test)** | **This model** | **2,102** | 57 | 21 |
| | Recent form (last 4 gameweeks) | 1,928 | 75 | 39 |
| | FPL's published forecast | 1,744 | 90 | 54 |
| 2023-24 (validation) | This model | 2,200 | 64 | 28 |
| | Recent form | 1,804 | 91 | 55 |
| | FPL's published forecast | 1,748 | 102 | 66 |

Bootstrapping the 37 gameweeks of the held-out season:

| 2024-25 comparison | Season-total gap | 95% interval | Gameweeks ahead |
|---|---|---|---|
| This model − FPL's forecast | +358 | +151 to +550 | 29 of 37 |
| This model − recent form | +174 | −29 to +370 | 24 of 37 |

**Reading it:**
- **Against FPL's published forecast the margin is significant:** +358 points over a
  season, ahead in 29 of 37 gameweeks.
- **Against recent form the model is ahead but the interval just includes zero.** One
  season of FPL points is noisy, because squads diverge and luck compounds. In the
  validation season the gap was +396.
- **Noisier forecasts make worse transfer decisions.** Form and FPL's forecast take two
  to three times as many hits, chasing last week's points.

# Still to come

- **Market benchmark.** De-margined closing odds from football-data.co.uk are the real bar.
  Closing prices include team news these models never see, so expect the market to win.
  The useful question is by how much.
- **Multi-season validation** on the longer history, replacing the single-season tuning.
- **Stacking and recalibration** (isotonic or Platt) of the two models.

# Limitations

- **No team news.** No lineups, injuries, suspensions or transfers. The market sees all
  of these.
- **Scorelines only.** Shot-based expected goals would separate luck from skill faster.
- **Promoted clubs start as the average of the three weakest clubs.** A crude prior:
  early-season predictions for promoted sides are the weakest part of the model.
