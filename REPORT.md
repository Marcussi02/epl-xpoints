# Results

Two questions, answered without letting any model see the future:

1. **Match prediction:** how well can a model built only from past scorelines predict
   Premier League matches?
2. **Fantasy Premier League:** do those predictions make better FPL decisions?

Reproduce everything here with:

```bash
epl tune       # match models: validation seasons 2016-17 to 2021-22 only
epl backtest   # match models: test seasons 2022-23 to 2025-26, with the market
epl xp         # FPL expected points vs FPL's own forecast, 2024-25
epl season     # FPL season replay, 2024-25
```

# Part 1: match prediction

## Method

- **Data:** every Premier League result and closing bookmaker odds from 2000-01 to
  2025-26 (9,880 matches), from football-data.co.uk.
- **Walk-forward:** for each week of a test season, every model is refitted on matches that
  finished before that week, then predicts the week's fixtures from pre-match information
  only. The harness raises an error if any training match overlaps the week being predicted.
- **Split:**
  - 2000-01 to 2015-16: history only.
  - 2016-17 to 2021-22: validation; hyper-parameters are chosen here, by mean RPS over
    all six seasons (2,280 matches).
  - 2022-23 to 2025-26: test; 1,520 matches, looked at once with frozen settings.
- **The market:** closing odds (the average across bookmakers where recorded, else
  Pinnacle's closing price, else pre-match odds), with the bookmaker's margin removed by
  the power method. Closing prices include team news, lineups and everything
  the betting market knows at kick-off, so they are the bar no public model is expected
  to clear.
- **Metrics:**
  - Ranked probability score (RPS, the standard for ordered football outcomes).
  - Log loss and Brier score.
  - Accuracy (for intuition only).
  - Expected calibration error.
- **Significance:** the paired bootstrap resamples whole weeks of matches, so a gap counts
  as real only when its 95% interval excludes zero.

## Choosing settings on six validation seasons

| Model | Grid | Chosen | Validation RPS |
|---|---|---|---|
| Elo | K ∈ {10, 15, 20, 25, 30} × season regression ∈ {0, 0.1, 0.2, 0.33} | K = 10, regression 0.1 | 0.1980 |
| Dixon–Coles | ξ ∈ {0.0006, 0.001, 0.0019, 0.003} × ridge ∈ {0, 1, 2, 5} | ξ = 0.0019, ridge 2 | 0.1968 |

- **Dixon–Coles is flat near the top:** the best eight settings sit within 0.0003 RPS of
  each other. Dropping the ridge penalty entirely is the only clearly worse choice.
- **ξ = 0.0019 is the value from the original paper:** a half-life of about a year.

An earlier version had only 2019-20 onwards, tuned on the single season 2021-22, and
scored Elo 0.2057 and Dixon–Coles 0.2045 on the test seasons; its tuned Elo did worse
than its own untuned defaults. With 26 seasons of data and settings chosen on six, both
models improve on the test seasons (below). Part of that comes from the longer history
itself, not only from better settings.

## Held-out results (2022-23 to 2025-26)

| Model | RPS ↓ | Log loss ↓ | Brier ↓ | Accuracy | Calibration error ↓ |
|---|---|---|---|---|---|
| Frequency (no skill) | 0.2316 | 1.068 | 0.646 | 44.5% | 0.009 |
| Elo + ordered logit | 0.2039 | 0.988 | 0.590 | 52.4% | 0.013 |
| **Dixon–Coles** | **0.2030** | **0.986** | **0.587** | 52.6% | **0.007** |
| Bookmaker closing odds | 0.1947 | 0.960 | 0.570 | 55.1% | 0.007 |

RPS by season:

| Season | Frequency | Elo | Dixon–Coles | Market |
|---|---|---|---|---|
| 2022-23 | 0.2275 | **0.2062** | 0.2100 | 0.1974 |
| 2023-24 | 0.2342 | **0.1894** | 0.1905 | 0.1802 |
| 2024-25 | 0.2369 | 0.2112 | **0.2022** | 0.1962 |
| 2025-26 | 0.2278 | 0.2088 | 0.2093 | 0.2048 |

Paired differences in RPS (negative = first model better):

| Comparison | Mean difference | 95% interval |
|---|---|---|
| Elo − Frequency | −0.0277 | −0.0343 to −0.0211 |
| Dixon–Coles − Frequency | −0.0286 | −0.0354 to −0.0218 |
| Dixon–Coles − Elo | −0.0009 | −0.0033 to +0.0016 |
| Dixon–Coles − Market | +0.0083 | +0.0054 to +0.0113 |
| Elo − Market | +0.0092 | +0.0065 to +0.0120 |

## What the numbers say

- **Both models have real skill.** Each cuts RPS by about 12% against frequency-only
  guessing, and the interval is nowhere near zero.
- **The market is better, by a measurable margin.** Dixon–Coles closes 78% of the gap
  between knowing nothing and the closing odds, from scorelines alone. The remaining
  0.008 RPS is significant, and it is what team news, lineups and the market's other
  information are worth. The market is better in every test season, by 0.004 to 0.013.
- **Dixon–Coles is as well calibrated as the market.** Its calibration error is 0.007, the
  same as the closing odds; Elo's is about twice that.
- **Dixon–Coles and Elo are statistically tied.** Dixon–Coles is still the one to build
  on: it predicts whole scorelines, which the FPL optimiser
  needs for clean-sheet and goal expectations.
- **Elo's home-advantage setting barely matters.** The ordered-logit link learns home
  advantage from data, absorbing it into its thresholds.
- **2020-21 is an outlier worth knowing about.** Matches were played without crowds, and
  it is the only season where away wins (40.3%) outnumbered home wins (37.9%). It sits in
  the validation seasons, and in training its weight fades with the time decay.

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
- **The match model inside FPL keeps its own settings** (ξ = 0.0006). The six-season
  match tuning in Part 1 prefers ξ = 0.0019. Swapped in here, it leaves expected-points
  accuracy unchanged (MAE 1.011 vs 1.012), but the 2023-24 validation replay scores 139
  points fewer, so the FPL pipeline stays with the setting it was tuned with. For
  completeness, the 2024-25 test replay with ξ = 0.0019 scores 2,034 (−68), still ahead
  of both other forecasts.

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

- **Stacking and recalibration** (isotonic or Platt) of the two models.

# Limitations

- **No team news.** No lineups, injuries, suspensions or transfers. The market sees all
  of these.
- **Scorelines only.** Shot-based expected goals would separate luck from skill faster.
- **Promoted clubs start as the average of the three weakest clubs.** A crude prior:
  early-season predictions for promoted sides are the weakest part of the model.
