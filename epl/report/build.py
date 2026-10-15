"""Render the weekly static site: one self-contained HTML page, no scripts or CDNs.

    this week    match probabilities and expected goals, the best squad from scratch,
                 and the top players by expected points
    track record held-out backtest scores against the bookmaker market, and a
                 reliability diagram for Dixon-Coles

Colours follow one rule set: home and away are the two poles of a diverging pair
(blue, red) with a neutral grey draw, every segment is labelled with its number, and
dark mode has its own validated steps.
"""

import html
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment
from markupsafe import Markup

CSS = """
:root { color-scheme: light; --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#6e6c67; --grid:#e1e0d9; --axis:#c3c2b7; --home:#2a78d6; --away:#e34948;
  --draw:#c3c2b7; --draw-ink:#0b0b0b; --ring:rgba(11,11,11,.1); --accent:#2a78d6; }
@media (prefers-color-scheme: dark) { :root { color-scheme: dark; --surface:#1a1a19;
  --page:#0d0d0d; --ink:#fff; --ink2:#c3c2b7; --muted:#9a9890; --grid:#2c2c2a; --axis:#383835;
  --home:#3987e5; --away:#e66767; --draw:#52514e; --draw-ink:#fff; --ring:rgba(255,255,255,.1);
  --accent:#3987e5; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
main { max-width:1040px; margin:0 auto; padding:28px 16px 56px; }
h1 { font-size:24px; margin:0; } h2 { font-size:17px; margin:0 0 4px; }
.sub,.note { color:var(--ink2); } .note { font-size:13px; margin:2px 0 12px; }
.card { background:var(--surface); border-radius:12px; box-shadow:0 0 0 1px var(--ring);
  padding:18px; margin-top:16px; overflow-x:auto; }
table { width:100%; border-collapse:collapse; font-size:14px; }
th,td { text-align:left; padding:8px 8px 8px 0; border-bottom:1px solid var(--grid);
  vertical-align:middle; }
th { color:var(--ink2); font-weight:500; font-size:13px; }
td.num,th.num { text-align:right; font-variant-numeric:tabular-nums; }
.bar { display:flex; gap:2px; height:22px; min-width:220px; border-radius:4px; overflow:hidden; }
.bar span { display:flex; align-items:center; justify-content:center; font-size:12px;
  font-variant-numeric:tabular-nums; color:#fff; white-space:nowrap; }
.bar .h { background:var(--home); } .bar .a { background:var(--away); }
.bar .d { background:var(--draw); color:var(--draw-ink); }
.legend { display:flex; gap:16px; font-size:13px; color:var(--ink2); margin-bottom:8px; }
.key { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.grid2 > * { min-width:0; } .grid2 > table { align-self:start; }
.pill { display:inline-block; font-size:11px; padding:1px 6px; border-radius:999px;
  box-shadow:inset 0 0 0 1px var(--ring); color:var(--ink2); margin-left:6px; }
.best { font-weight:600; }
footer { margin-top:28px; font-size:12px; color:var(--muted); }
footer a { color:var(--ink2); }
@media (max-width:760px) { .grid2 { grid-template-columns:1fr; } }
@media (max-width:560px) { .wide { display:none; } .bar { min-width:120px; }
  .card { padding:14px; } th,td { padding-right:6px; } }
"""

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EPL xPoints · Gameweek {{ gw }}</title>
<meta name="description" content="Premier League match predictions and FPL picks for gameweek {{ gw }}.">
<style>{{ css }}</style></head>
<body><main>
<header>
  <h1>Gameweek {{ gw }} predictions</h1>
  <p class="sub">Season {{ season }} · deadline {{ deadline }} UTC · generated {{ generated }} UTC</p>
</header>

<section class="card">
  <h2>Matches</h2>
  <p class="note">Dixon–Coles probabilities from results only: no team news, no odds.</p>
  <div class="legend"><span><i class="key" style="background:var(--home)"></i>Home win</span>
    <span><i class="key" style="background:var(--draw)"></i>Draw</span>
    <span><i class="key" style="background:var(--away)"></i>Away win</span></div>
  <table>
    <thead><tr><th class="wide">Kick-off</th><th>Match</th><th>Probabilities</th>
      <th class="num wide">Expected goals</th><th class="num">Likeliest score</th></tr></thead>
    <tbody>{% for m in matches %}
    <tr><td class="wide">{{ m.kickoff }}</td><td>{{ m.home }} v {{ m.away }}</td>
      <td><div class="bar" role="img" aria-label="Home {{ m.ph }}%, draw {{ m.pd }}%, away {{ m.pa }}%">
        <span class="h" style="flex:{{ m.ph }}">{% if m.ph >= 7 %}{{ m.ph }}%{% endif %}</span>
        <span class="d" style="flex:{{ m.pd }}">{% if m.pd >= 7 %}{{ m.pd }}%{% endif %}</span>
        <span class="a" style="flex:{{ m.pa }}">{% if m.pa >= 7 %}{{ m.pa }}%{% endif %}</span></div></td>
      <td class="num wide">{{ m.xgh }} – {{ m.xga }}</td><td class="num">{{ m.score }}</td></tr>
    {% endfor %}</tbody>
  </table>
</section>

<div class="grid2">
<section class="card">
  <h2>Best squad from scratch</h2>
  <p class="note">£100m, all FPL squad rules, solved exactly as an integer program.
    Expected {{ squad_points }} points this gameweek with the captain doubled.</p>
  <table><thead><tr><th>Player</th><th>Club</th><th class="num">Price</th>
    <th class="num">xPts</th></tr></thead><tbody>
  {% for p in xi %}<tr><td>{{ p.name }}{% if p.captain %}<span class="pill">C</span>{% elif p.vice %}<span class="pill">V</span>{% endif %}
    <span class="pill">{{ p.position }}</span></td><td>{{ p.team }}</td>
    <td class="num">£{{ p.price }}m</td><td class="num">{{ p.xp }}</td></tr>{% endfor %}
  <tr><th colspan="4">Bench</th></tr>
  {% for p in bench %}<tr><td>{{ p.name }}<span class="pill">{{ p.position }}</span></td>
    <td>{{ p.team }}</td><td class="num">£{{ p.price }}m</td><td class="num">{{ p.xp }}</td></tr>{% endfor %}
  </tbody></table>
</section>

<section class="card">
  <h2>Top players by expected points</h2>
  <p class="note">This gameweek, scaled by FPL's injury and suspension flags.</p>
  <table><thead><tr><th>Player</th><th>Club</th><th class="num">Price</th>
    <th class="num">xPts</th></tr></thead><tbody>
  {% for p in top %}<tr><td>{{ p.name }}<span class="pill">{{ p.position }}</span>{% if p.flag %}<span class="pill">{{ p.flag }}</span>{% endif %}</td>
    <td>{{ p.team }}</td><td class="num">£{{ p.price }}m</td><td class="num">{{ p.xp }}</td></tr>{% endfor %}
  </tbody></table>
</section>
</div>

{% if record %}
<section class="card">
  <h2>Track record</h2>
  <p class="note">Walk-forward backtest on held-out seasons {{ record_seasons }} ({{ record_matches }} matches):
    every prediction made using only matches played before it. Lower is better.</p>
  <div class="grid2">
  <table><thead><tr><th>Model</th><th class="num">RPS</th><th class="num">Log loss</th>
    <th class="num">Accuracy</th></tr></thead><tbody>
  {% for r in record %}<tr{% if r.best %} class="best"{% endif %}><td>{{ r.model }}</td><td class="num">{{ r.rps }}</td>
    <td class="num">{{ r.log_loss }}</td><td class="num">{{ r.accuracy }}</td></tr>{% endfor %}
  </tbody></table>
  <div>{{ reliability_svg }}</div>
  </div>
</section>
{% endif %}

<footer>Data: football-data.co.uk results and odds; the official FPL API; the
<a href="https://github.com/vaastav/Fantasy-Premier-League">vaastav/Fantasy-Premier-League</a>
dataset (MIT). Independent project, not affiliated with the Premier League or FPL.
<a href="https://github.com/Marcussi02/epl-xpoints">Source and method</a>.</footer>
</main></body></html>
"""

MODEL_NAMES = {"market": "Bookmaker closing odds", "dixon_coles": "Dixon–Coles",
               "elo": "Elo", "frequency": "Historical frequencies"}


def reliability_svg(table: pd.DataFrame, width: int = 420, height: int = 300) -> str:
    """Reliability diagram: forecast probability vs observed frequency, one dot per bin
    (area ~ matches), 95% interval whiskers, and the perfect-calibration diagonal."""
    pad_l, pad_b, pad_t, pad_r = 56, 36, 12, 20
    w, h = width - pad_l - pad_r, height - pad_t - pad_b

    def x(v):
        return pad_l + v * w

    def y(v):
        return pad_t + (1 - v) * h

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
             'aria-label="Reliability diagram for Dixon–Coles">']
    for t in (0, 0.25, 0.5, 0.75, 1):
        parts.append(f'<line x1="{x(0)}" x2="{x(1)}" y1="{y(t)}" y2="{y(t)}" '
                     'stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{y(t) + 4}" text-anchor="end" font-size="11" '
                     f'fill="var(--muted)">{t:.0%}</text>')
        parts.append(f'<text x="{x(t)}" y="{height - pad_b + 16}" text-anchor="middle" '
                     f'font-size="11" fill="var(--muted)">{t:.0%}</text>')
    parts.append(f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" '
                 'stroke="var(--axis)" stroke-width="1.5"/>')
    n_max = table["n"].max()
    for r in table.itertuples():
        cx, cy = x(r.mean_forecast), y(r.observed)
        radius = 4 + 6 * np.sqrt(r.n / n_max)
        parts.append(f'<line x1="{cx}" x2="{cx}" y1="{y(r.ci_low)}" y2="{y(r.ci_high)}" '
                     'stroke="var(--accent)" stroke-width="2" stroke-linecap="round"/>')
        tip = (f"Forecast {r.mean_forecast:.0%}, happened {r.observed:.0%} "
               f"({int(r.n)} outcomes)")
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{radius:.1f}" fill="var(--accent)" '
                     f'stroke="var(--surface)" stroke-width="2"><title>{html.escape(tip)}</title>'
                     '</circle>')
    parts.append(f'<text x="{pad_l + w / 2}" y="{height - 4}" text-anchor="middle" font-size="12" '
                 'fill="var(--ink2)">Forecast probability</text>')
    parts.append(f'<text transform="translate(12 {pad_t + h / 2}) rotate(-90)" '
                 'text-anchor="middle" font-size="12" fill="var(--ink2)">How often it happened'
                 '</text></svg>')
    return "".join(parts)


def _pct(values: np.ndarray) -> list[int]:
    """Whole percentages that still add up to 100."""
    raw = np.asarray(values) * 100
    out = np.floor(raw).astype(int)
    for i in np.argsort(raw - out)[::-1][: 100 - out.sum()]:
        out[i] += 1
    return out.tolist()


def render(gw: int, season: str, deadline: pd.Timestamp, matches: pd.DataFrame,
           players: pd.DataFrame, xp: pd.Series, plan, record: pd.DataFrame | None = None,
           reliability: pd.DataFrame | None = None, record_seasons: str = "") -> str:
    def row(e: int) -> dict:
        p = players.loc[e]
        return {"name": p["name"], "team": p["team"], "position": p["position"],
                "price": f"{p['price']:.1f}", "xp": f"{xp.get(e, 0.0):.1f}",
                "captain": e == plan.captain, "vice": e == plan.vice}

    match_rows = []
    for m in matches.itertuples():
        ph, pd_, pa = _pct([m.p_home, m.p_draw, m.p_away])
        match_rows.append({"kickoff": m.kickoff.strftime("%a %d %b %H:%M"), "home": m.home,
                           "away": m.away, "ph": ph, "pd": pd_, "pa": pa,
                           "xgh": f"{m.xg_home:.1f}", "xga": f"{m.xg_away:.1f}",
                           "score": m.likely_score})
    top = []
    for e in xp.sort_values(ascending=False).head(15).index:
        r = row(e)
        chance = players.loc[e, "chance_next"]
        r["flag"] = f"{chance:.0f}% to play" if chance < 100 else ""
        top.append(r)
    rec = None
    if record is not None:
        best = record["rps"].min()
        rec = [{"model": MODEL_NAMES.get(r.model, r.model), "rps": f"{r.rps:.4f}",
                "log_loss": f"{r.log_loss:.3f}", "accuracy": f"{r.accuracy:.1%}",
                "best": r.rps == best} for r in record.sort_values("rps").itertuples()]
    env = Environment(autoescape=True)
    return env.from_string(TEMPLATE).render(
        css=Markup(CSS), gw=gw, season=season, deadline=deadline.strftime("%a %d %b %H:%M"),
        generated=datetime.now(UTC).strftime("%d %b %Y %H:%M"), matches=match_rows,
        xi=[row(e) for e in plan.xi], bench=[row(e) for e in plan.bench],
        squad_points=f"{plan.expected_points:.1f}", top=top, record=rec,
        record_matches=int(record["matches"].iloc[0]) if record is not None else 0,
        record_seasons=record_seasons,
        reliability_svg=Markup(reliability_svg(reliability)) if reliability is not None else "",
    )


def write(path: Path, page: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path
