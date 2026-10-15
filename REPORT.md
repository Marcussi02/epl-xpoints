# Results

How well can a model built only from past scorelines predict Premier League matches?
This report covers milestone 3: two models, tuned on one season and scored on four
seasons they never saw.

Reproduce everything here with:

```bash
epl tune       # validation season 2021-22 only
epl backtest   # test seasons 2022-23 to 2025-26
```

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

## Still to come

- **Market benchmark.** De-margined closing odds from football-data.co.uk are the real bar.
  Published studies put market RPS for the Premier League a little below these models,
  so expect the market to win. The useful question is by how much.
- **Multi-season validation** on the longer history, replacing the single-season tuning.
- **Stacking and recalibration** (isotonic or Platt) of the two models.

## Limitations

- **No team news.** No lineups, injuries, suspensions or transfers. The market sees all
  of these.
- **Scorelines only.** Shot-based expected goals would separate luck from skill faster.
- **Promoted clubs start as the average of the three weakest clubs.** A crude prior:
  early-season predictions for promoted sides are the weakest part of the model.
