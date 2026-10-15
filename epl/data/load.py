"""Raw downloads -> tidy, typed tables.

    matches   one row per Premier League match (2000-01 onwards):
              match_id, season, date, home, away, hg, ag, result,
              odds_h, odds_d, odds_a, odds_source

Odds are the most efficient price available for that season, best first:
average closing (AvgC*, from 2019-20), Pinnacle closing (PSC*, from 2012-13),
market average before kick-off (Avg*, BbAv*), then Bet365 (B365*). The source is
recorded per row, so an analysis can restrict itself to genuine closing prices.

Only the columns in PREMATCH_COLUMNS are known before kick-off. The backtest hands
models nothing else, so no model can peek at a result it is asked to predict.
"""

import csv
import io
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import download
from .teams import canonical

PREMATCH_COLUMNS = ["match_id", "season", "date", "home", "away", "odds_h", "odds_d", "odds_a"]
RESULT_COLUMNS = ["hg", "ag", "result"]
OUTCOMES = ("H", "D", "A")

# (home, draw, away) column triples, most efficient first.
ODDS_PRIORITY = [
    ("avg_closing", ("AvgCH", "AvgCD", "AvgCA")),
    ("pinnacle_closing", ("PSCH", "PSCD", "PSCA")),
    ("avg", ("AvgH", "AvgD", "AvgA")),
    ("betbrain_avg", ("BbAvH", "BbAvD", "BbAvA")),
    ("bet365", ("B365H", "B365D", "B365A")),
]


def parse_football_data(raw: bytes, season: str) -> pd.DataFrame:
    """Parse one football-data.co.uk E0.csv into the tidy match schema."""
    text = raw.decode("utf-8-sig", errors="replace")
    # Some seasons have rows with more fields than the header (stray trailing columns).
    # Trim every row to the header instead of skipping it, or whole matches go missing.
    rows = [r for r in csv.reader(io.StringIO(text)) if any(cell.strip() for cell in r)]
    if not rows:
        return _empty()
    header = rows[0]
    body = [(r + [""] * len(header))[: len(header)] for r in rows[1:]]
    df = pd.DataFrame(body, columns=header).replace("", np.nan)
    for col in ("FTHG", "FTAG"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])  # trailing blank rows
    if df.empty:
        return _empty()

    out = pd.DataFrame({
        "season": season,
        "date": pd.to_datetime(df["Date"], dayfirst=True, format="mixed"),
        "home": df["HomeTeam"].map(canonical),
        "away": df["AwayTeam"].map(canonical),
        "hg": df["FTHG"].astype(int),
        "ag": df["FTAG"].astype(int),
    })
    out["result"] = np.select([out.hg > out.ag, out.hg == out.ag], ["H", "D"], "A")

    out["odds_source"] = None
    for col in ("odds_h", "odds_d", "odds_a"):
        out[col] = np.nan
    for source, cols in ODDS_PRIORITY:
        if not set(cols) <= set(df.columns):
            continue
        prices = df[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy()
        usable = np.isfinite(prices).all(axis=1) & (prices > 1).all(axis=1)
        fill = usable & out["odds_source"].isna().to_numpy()
        out.loc[fill, ["odds_h", "odds_d", "odds_a"]] = prices[fill]
        out.loc[fill, "odds_source"] = source

    out["match_id"] = (
        out["date"].dt.strftime("%Y%m%d") + "-" + out["home"] + "-" + out["away"]
    ).str.replace(" ", "_")
    cols = ["match_id", "season", "date", "home", "away", *RESULT_COLUMNS,
            "odds_h", "odds_d", "odds_a", "odds_source"]
    return out[cols].sort_values(["date", "home"]).reset_index(drop=True)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["match_id", "season", "date", "home", "away", *RESULT_COLUMNS,
                                 "odds_h", "odds_d", "odds_a", "odds_source"])


def seasons(first: str, last: str) -> list[str]:
    """['2019-20', '2020-21', ...] inclusive."""
    start, stop = int(first[:4]), int(last[:4])
    return [f"{y}-{(y + 1) % 100:02d}" for y in range(start, stop + 1)]


def load_matches(
    season_list: list[str], cache_dir: Path = download.CACHE_DIR, fetch=download.get_bytes
) -> pd.DataFrame:
    frames = [
        parse_football_data(download.football_data_csv(s, cache_dir, fetch=fetch), s)
        for s in season_list
    ]
    matches = pd.concat(frames, ignore_index=True)
    validate(matches)
    return matches


def validate(matches: pd.DataFrame) -> None:
    """Fail loudly on data that would quietly corrupt a backtest."""
    dupes = matches["match_id"].duplicated()
    if dupes.any():
        raise ValueError(f"duplicate fixtures: {matches.loc[dupes, 'match_id'].tolist()[:5]}")
    if (matches["home"] == matches["away"]).any():
        raise ValueError("a team cannot play itself")
    odds = matches[["odds_h", "odds_d", "odds_a"]].dropna()  # some sources carry no odds
    implied = (1 / odds).sum(axis=1)
    bad = implied[(implied < 1.0) | (implied > 1.15)]
    if not bad.empty:
        raise ValueError(
            f"{len(bad)} rows have implausible odds overrounds, e.g. {bad.iloc[0]:.3f}"
        )


def complete_seasons(matches: pd.DataFrame) -> list[str]:
    counts = matches.groupby("season").size()
    return sorted(counts[counts == 380].index)


def save(matches: pd.DataFrame, path: Path = Path("data/processed/matches.parquet")) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(path, index=False)
    return path


def query(sql: str, **tables: pd.DataFrame) -> pd.DataFrame:
    """Run DuckDB SQL over in-memory frames, e.g. query("SELECT ... FROM m", m=matches)."""
    con = duckdb.connect()
    for name, frame in tables.items():
        con.register(name, frame)
    return con.execute(sql).df()
