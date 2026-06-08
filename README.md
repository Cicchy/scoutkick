# ScoutKick

Expected Points Added (EPA) rating system for FIRST Tech Challenge (FTC), ported from [Statbotics](https://statbotics.io).

Pulls match data from `api.ftcscout.org`, computes EPA ratings via an EWMA learning loop, and serves them through a FastAPI REST API and interactive CLI.

## Quick Start

```bash
# From repo root
$env:PYTHONPATH="."
python scoutkick/backend/main.py          # Start API server (http://127.0.0.1:8000)
python scoutkick/scout.py                 # Interactive CLI shell
```

The first run of either command triggers the pipeline to fetch and process matches. Cached data is stored in `cache/epa_data.db`.

## CLI

```
> team 26914         Show EPA breakdown with match history
> compare 26914 32736  Head-to-head comparison
> top 20              Leaderboard
> search 269          Find teams
> stats               Distribution statistics
```

## API

All endpoints return `{"value": [...], "count": N}` (except `/v1/seasons` / single-resource endpoints).  
Docs at `http://127.0.0.1:8000/docs`.

### Season

| Endpoint | Description |
|----------|-------------|
| `GET /v1/seasons` | List cached seasons |
| `GET /v1/season/{season}` | Season metadata (score_mean, score_sd, num_matches, num_teams) |

### Teams

| Endpoint | Description |
|----------|-------------|
| `GET /v1/teams?season=2025&metric=norm_epa&limit=50` | Paginated team list, sorted by any metric |
| `GET /v1/team/{team}?season=2025` | Team season stats with `team_matches` array (EPA evolution) |
| `GET /v1/team/{team}/events?season=2025` | Per-event EPA stats for a team |
| `GET /v1/team/{team}/matches?season=2025` | Paginated per-match EPA history |
| `GET /v1/team/{team}/evolution` | Cross-season EPA summary (all seasons) |

The `GET /v1/team/{team}` response includes a `team_matches` array ordered chronologically — frontends use this for the EPA-over-time graph:

```json
{
  "team": 26914,
  "season": "2025",
  "total": 85.3,
  "auto": 25.0,
  "teleop": 40.3,
  "endgame": 20.0,
  "norm_epa": 1520.0,
  "count": 18,
  "team_matches": [
    {
      "event_code": "USNEBEL",
      "match_id": "1",
      "epa_pre": 80.0,
      "epa_post": 82.5,
      "win_prob": 0.65,
      "is_elim": false,
      "processed_at": "2025-01-15 10:30:00"
    }
  ]
}
```

### Events

| Endpoint | Description |
|----------|-------------|
| `GET /v1/events?season=2025` | All events with aggregate EPA stats |
| `GET /v1/event/{code}?season=2025` | Event detail with team list |
| `GET /v1/event/{code}/matches?season=2025` | Match list within an event |

### Matches

| Endpoint | Description |
|----------|-------------|
| `GET /v1/matches?season=2025` | Global match list (filterable by team, event, elim) |
| `GET /v1/match/{event}/{match}?season=2025` | Single match detail with all teams |

### Predict

| Endpoint | Description |
|----------|-------------|
| `GET /v1/predict?red=26914,32736&blue=23400,24599` | Predict match outcome + component scores |
| `GET /v1/compare?teams=26914,32736,23400` | Side-by-side team comparison |

## How It Works

1. **Fetch** — Pulls match data from `api.ftcscout.org/graphql` using per-season GraphQL fragments.
2. **Clean** — `BaseCleaner` subclasses map raw alliance scores to a 32-dim EPA vector. 7 seasons supported (2019–2025).
3. **Learn** — Runs all matches through an EWMA loop. Each team is a `SkewNormal` distribution. After each match, prediction error is split equally across alliance teammates and used to update ratings.
4. **Calibrate** — Computes `score_sd` and component means from the dataset.
5. **Serve** — Results stored in SQLite, served via FastAPI.

### Vector dimensions

Seasons 2019–2023 use 4 dimensions: `[total, auto, teleop, endgame]`.  
2024 uses 10.  
2025 uses 10 active dimensions in a 32-slot vector:

| Index | Component |
|-------|-----------|
| 0 | Total EPA (`preFoulTotal`) |
| 1 | Auto EPA |
| 2 | Teleop EPA |
| 3 | Endgame EPA |
| 4 | RP1 (Movement) |
| 5 | RP2 (Goal) |
| 6 | RP3 (Pattern) |
| 7 | Auto Classified |
| 8 | Teleop Classified |
| 9 | Teleop Depot |

### Key parameters

- **Learning rate**: `alpha = 1 / (1 + n * 0.1)` — decays with each match
- **Attribution**: Error split equally across 2 alliance teammates
- **Elim match weight**: 0.33 (33%)
- **Default initial mean**: 20.0 (total), variance: 100.0

## Project Structure

```
scoutkick/
├── backend/
│   ├── main.py              # FastAPI entry point
│   └── src/
│       ├── core/
│       │   ├── math.py       # SkewNormal distribution, unit_sigmoid
│       │   └── config.py     # Per-season configs (dimensions, defaults)
│       ├── data/
│       │   ├── cleaner.py    # Season-specific ETL (CleanerRegistry)
│       │   ├── ftcscout_api.py
│       │   └── read_ftcscout.py
│       ├── services/
│       │   ├── epa_service.py    # EPAEngine (prediction + update loop)
│       │   ├── pipeline_service.py  # EPAPipeline (orchestrator)
│       │   ├── calibrate.py
│       │   └── init_epa.py
│       ├── storage/
│       │   └── sqlite_storage.py
│       └── api/
│           ├── router.py
│           ├── deps.py
│           ├── season.py
│           ├── team.py
│           ├── event.py
│           ├── match.py
│           └── predict.py
├── tests/
│   └── test_engine.py
├── scout.py              # Interactive CLI
└── CLAUDE.md
```

## Client Package

A zero-dependency Python client is available at `scoutkick-python/` and on PyPI as `scoutkick-api`:

```python
from scoutkick_api import ScoutKick
sk = ScoutKick()
sk.get_team(26914)
sk.predict(red=[26914, 32736], blue=[23400, 24599])
```

## Data Source

All match data sourced from [FTC Scout](https://ftcscout.org) via their public GraphQL API (`api.ftcscout.org/graphql`). Cached in `cache/ftcscout/*.p` to minimize repeated fetches.
