# ScoutKick

![ScoutKick](logo.png)

[![PyPI](https://img.shields.io/pypi/v/scoutkick-api)](https://pypi.org/project/scoutkick-api/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)

Expected Points Added (EPA) rating system for FIRST Tech Challenge (FTC), ported from [Statbotics](https://statbotics.io). ScoutKick pulls match data from FTC Scout, computes EPA ratings via an EWMA learning loop, and exposes them through a REST API and Python package.

---

## Python API

```bash
pip install scoutkick-api
```

```python
from scoutkick_api import ScoutKick

sk = ScoutKick()

print(sk.get_team(26914))
print(sk.predict(red=[26914, 32736], blue=[23400, 24599]))
print(sk.compare(teams=[26914, 32736, 23400]))

# All methods: get_seasons, get_teams, get_team, get_events,
# get_event, get_matches, get_match, predict, compare,
# get_clusters, get_complementarity, get_alliance_partners,
# get_trajectory_clusters, run_pipeline, get_pipeline_status
```

---

## REST API

Live at [`scoutkick.onrender.com`](https://scoutkick.onrender.com) — docs at `/docs`.

| Endpoint | Description |
|----------|-------------|
| `GET /v1/teams?season=2025` | List teams |
| `GET /v1/team/{team}?season=2025` | Team EPA + match history |
| `GET /v1/events?season=2025` | List events |
| `GET /v1/event/{code}?season=2025` | Event detail |
| `GET /v1/matches?season=2025` | List matches |
| `GET /v1/match/{event}/{match}?season=2025` | Match detail |
| `GET /v1/predict?red=26914,32736&blue=23400,24599` | Predict match |
| `GET /v1/compare?teams=26914,32736` | Compare teams |

---

## Local server

```bash
$env:PYTHONPATH="."
pip install -r backend/requirements.txt
python backend/main.py
```

Data is cached in `backend/cache/`. Populate with `POST /v1/data/run?season=2025`.

---

## How it works

1. **Fetch** — Match data from FTC Scout GraphQL API
2. **Clean** — Map scores to a 32-dim EPA vector (7 seasons: 2019–2025)
3. **Learn** — EWMA loop. Each team is a SkewNormal distribution. Prediction error updates mean/variance/skewness
4. **Calibrate** — Compute score baseline from match data
5. **Serve** — Results stored in SQLite/PostgreSQL, served via FastAPI

---

## License

[MIT](LICENSE)
