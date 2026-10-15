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

**A leaky benchmark, caught.** The community dataset records FPL's own expected
points (`xP`) for every player and gameweek, and at first it looked unbeatable: a
Spearman correlation of 0.58 with the gameweek's points among players who played.
Consecutive gameweeks' points correlate only 0.21, though, and the same `xP` correlates
just 0.22 with the *next* gameweek's points. So the value on a gameweek's row was
recorded after that gameweek: it is FPL's forecast for the following one, built from
form that already includes the row's result. Shifted to the gameweek it was published
for, it becomes a fair benchmark.

Held-out 2024-25, compared on the same 25,159 player-gameweeks:

| Forecast | MAE ↓ | Rank correlation ↑ | Rank correlation, players who played ↑ | Top 50 picks per gameweek: actual points ↑ | Bias |
|---|---|---|---|---|---|
| **This model** | **1.023** | **0.704** | **0.351** | **4.30** | +0.04 |
| FPL's published forecast | 1.142 | 0.581 | 0.251 | 3.90 | −0.14 |

The validation season tells the same story (rank correlation among players who played:
0.342 vs 0.265). FPL's forecast also knows injury and suspension news, which the
historical files don't record. Live, the planner uses FPL's injury flags, so this
model gains that information too.

## Squad selection and transfers

- **The optimiser:** an exact integer program (HiGHS through `scipy.optimize.milp`).
  - Picks a 15-player squad (2 GK, 5 DEF, 5 MID, 3 FWD, at most 3 per club,
    £100m), a valid XI and a captain.
  - Plans transfers over a rolling horizon, with free transfers rolling over up to five.
  - Property tests generate random player pools and check that every plan obeys every rule.
- **The replay:** gameweeks 2–38 of a past season, with each week's decisions made at
  the deadline. Real points are counted with captaincy, auto-substitutions, hits and
  FPL selling prices. Chips are not used.
- **Tuning on 2023-24:** charging the planner 8 points per hit instead of the real 4
  cut hits from 37 to 15 and raised the season total. The 3- and 1-gameweek horizons
  tied, so the 3-gameweek horizon stays.

Held-out 2024-25, same frozen planner for every forecast:

| Forecast | Season points | Per gameweek | Transfers | Hits |
|---|---|---|---|---|
| **This model** (3-gameweek horizon) | **2,071** | **56.0** | 44 | 8 |
| Recent form (last 4 gameweeks) | 2,003 | 54.1 | 43 | 7 |
| FPL's published forecast | 1,966 | 53.1 | 43 | 7 |

**Honest reading:**
- The model finishes ahead in both seasons replayed. In 2023-24 (validation) it scored
  2,002 against 1,823 for form and 1,748 for FPL's forecast, with an untuned planner
  for the baselines.
- One season is not enough to call it, though. Bootstrapping the 37 gameweeks gives
  95% intervals on the season-total gap of −131 to +257 points against form and −93 to
  +317 against FPL. FPL points are noisy and squads diverge, so luck compounds.
- The player-level comparison above, across 25,000 forecasts, is the stronger evidence.
- The hit penalty mattered more to the baselines than to this model. Before tuning, on
  the same 2024-25 replay, the model scored 2,104, form 1,904 and FPL's forecast 1,744.
  Noisier forecasts make worse transfer decisions, and the penalty partly protects them.

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
