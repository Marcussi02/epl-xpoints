"""Polite, cached downloads from the public data sources.

    football-data.co.uk   one CSV per season: results plus pre-match and closing odds
    FPL API               players, prices, fixtures and per-gameweek history (unofficial)

Every response is cached on disk under data/cache/, so a backtest never re-downloads
history and the FPL API is hit at most once per refresh. Requests identify the client
and are spaced out; nothing here scrapes HTML.
"""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

USER_AGENT = "epl-xpoints/0.1 (+https://github.com/Marcussi02/epl-xpoints)"
CACHE_DIR = Path("data/cache")
FOOTBALL_DATA_URL = "https://football-data.co.uk/mmz4281/{code}/E0.csv"  # www. redirects here
FPL_API = "https://fantasy.premierleague.com/api/"
MIN_INTERVAL_S = 1.0

_last_request = 0.0


class FetchError(RuntimeError):
    """A URL could not be fetched after all retries."""


def season_code(season: str) -> str:
    """'2024-25' -> '2425', the form football-data.co.uk uses in its URLs."""
    start, end = season.split("-")
    if len(start) != 4 or len(end) != 2 or int(end) != (int(start) + 1) % 100:
        raise ValueError(f"season must look like '2024-25', got {season!r}")
    return start[2:] + end


def get_bytes(url: str, retries: int = 3, timeout: int = 30) -> bytes:
    global _last_request
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404):
                raise FetchError(f"GET {url} failed: HTTP {exc.code}") from exc
            last_error = exc
        except Exception as exc:  # network errors, timeouts
            last_error = exc
        if attempt < retries:
            time.sleep(2**attempt)
    raise FetchError(f"GET {url} failed after {retries} attempts: {last_error}")


def cached(url: str, path: Path, refresh: bool = False, fetch=get_bytes) -> bytes:
    """Return the cached copy of `url` at `path`, downloading it first if needed."""
    if path.exists() and not refresh:
        return path.read_bytes()
    data = fetch(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)  # never leave a half-written file in the cache
    return data


def football_data_csv(
    season: str, cache_dir: Path = CACHE_DIR, refresh: bool = False, fetch=get_bytes
) -> bytes:
    code = season_code(season)
    path = cache_dir / "football-data" / f"E0_{code}.csv"
    return cached(FOOTBALL_DATA_URL.format(code=code), path, refresh, fetch)


def fpl_json(
    endpoint: str, cache_dir: Path = CACHE_DIR, refresh: bool = True, fetch=get_bytes
) -> dict | list:
    """FPL API JSON, e.g. 'bootstrap-static/' or 'element-summary/302/'. Live data, so it
    refreshes by default; pass refresh=False to reuse the last snapshot."""
    name = endpoint.strip("/").replace("/", "_") or "root"
    path = cache_dir / "fpl" / f"{name}.json"
    return json.loads(cached(FPL_API + endpoint.lstrip("/"), path, refresh, fetch))
