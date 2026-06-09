# ScoutKick Deployment Guide

## Architecture

```
FastAPI app (backend.main:app)
  ├── Lifespan: runs pipeline for all seasons (2019→2025)
  │   └── Pipeline: fetch FTCScout data → calibrate → train EPA → save to SQLite
  ├── API: /v1/* endpoints (teams, seasons, predict, matches, events, clusters)
  └── Storage: SQLite via backend.src.storage.sqlite_storage.SQLiteStorage
```

## Local Development

### Setup
```bash
cd scoutkick/
$env:PYTHONPATH="."
pip install -r requirements.txt
```

### Run API Server
```bash
$env:PYTHONPATH="."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Server starts immediately; pipeline runs async in background for all 7 seasons (2019–2025). Check pipeline status at `GET /v1/pipeline`.

### Run Tests
```bash
$env:PYTHONPATH="."
python -m pytest tests/ -v --ignore=tests/test_clusters.py --ignore=tests/test_smoke.py
```

### Run Single-Engine Test
```bash
$env:PYTHONPATH="."
python tests/test_engine.py
```

### Run Pipeline Manually
```bash
$env:PYTHONPATH="."
python -c "
from backend.src.services.pipeline_service import EPAPipeline
p = EPAPipeline('2025')
p.run()
"
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONPATH` | `.` | Must be `.` (scoutkick root) for imports |
| `EPA_DB_PATH` | `cache/epa_data.db` (CWD-relative) | SQLite database path. Both pipeline and API use this. |
| `EPA_SEASONS` | All seasons (2019–2025) | Comma-separated, e.g. `2024,2025` |
| `FTCSCOUT_TIMEOUT` | `60` | GraphQL API timeout in seconds |

## Database Path (Critical)

**Both the pipeline and API must use the same database file.** Resolution order:

1. `EPA_DB_PATH` env var (if set)
2. `os.getcwd() + "/cache/epa_data.db"` (CWD-relative, default)

Relevant files:
- `backend/main.py:24` — pipeline startup
- `backend/src/api/deps.py:4` — API endpoints

## Files Changed (This Session)

| File | Change |
|------|--------|
| `backend/src/api/deps.py` | Use `EPA_DB_PATH` env var or CWD-relative default (was hardcoded 4-levels-up) |
| `backend/src/storage/sqlite_storage.py` | Create parent dir on init; add `load_all_seasons_meta()` |
| `backend/main.py` | Loop all seasons 2019→2025; track pipeline status via `/v1/pipeline` |
| `backend/src/api/season.py` | `/v1/seasons` returns all seasons (not just default) |
| `backend/src/services/pipeline_service.py` | Elim is only `Semis`/`Finals` (was anything != `Quals`) |
| `backend/src/api/match.py` | `is_elim` now included in per-team match detail |
| `backend/src/api/predict.py` | Predict dims from season config, not hardcoded 2025 list |
| `backend/src/data/ftcscout_api.py` | API timeout from `FTCSCOUT_TIMEOUT` env var (default 60s) |

## Match Type Weighting

| Tournament Level | Weight | Count Increments |
|-----------------|--------|-----------------|
| Quals | 1.0 | Yes |
| Semis | 0.33 | No |
| Finals | 0.33 | No |
| Practice / Scrimmage / LeagueMeet | 1.0 | Yes |

Weight blends old and new state: `weight * new_mean + (1-weight) * old_mean`.

## Global Chronological Ordering

Events are processed in **event_code string order** (not chronological). Cross-event match ordering is not guaranteed. Within an event, matches are sorted by `match_id`.

## Pipeline Details

- **7 seasons**: 2019 → 2020 → 2021 → 2022 → 2023 → 2024 → 2025
- **Cross-season carryover**: Each season loads prior seasons' EPA via `get_prior_seasons(look_back=4)` and `get_init_epa()`
- **First season (2019)**: all teams are rookies (no prior data)
- **Subsequent seasons**: returning teams initialized with prior EPA, rookies get `INIT_EPA = 1450`
- **Data source**: `api.ftcscout.org/graphql` (alliance-level scores, no per-robot breakdown)
- **GraphQL limit**: 100 events per season (`read_ftcscout.py:11`). If a season has >100 events, data is silently truncated.
- **Caching**: Raw GraphQL responses cached in `cache/ftcscout/*.p` (pickle). No cache invalidation — delete manually to refresh.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Root (includes pipeline status) |
| `GET /v1/pipeline` | Pipeline status & per-season results |
| `GET /v1/seasons` | All season metadata |
| `GET /v1/season/{season}` | Single season metadata |
| `GET /v1/teams?season=&metric=&ascending=&limit=&offset=` | List teams |
| `GET /v1/team/{team}?season=` | Team details + matches |
| `GET /v1/team/{team}/matches?season=&limit=&offset=` | Team matches |
| `GET /v1/events?season=&limit=&offset=` | List events |
| `GET /v1/event/{code}?season=` | Event details |
| `GET /v1/event/{code}/matches?season=` | Event matches |
| `GET /v1/matches?season=&event=&elim=&team=&metric=&ascending=&limit=&offset=` | List matches |
| `GET /v1/match/{event_code}/{match_id}?season=` | Match detail |
| `GET /v1/predict?red=...,...&blue=...,...&season=` | Match prediction |
| `GET /v1/compare?teams=...,...,...&season=` | Compare teams |
| `GET /v1/clusters?season=` | Team clusters |
| `GET /v1/team/{team}/playstyle?season=` | Team playstyle |
| `GET /v1/complementarity?teams=...,...&season=` | Team complementarity |
| `GET /v1/trajectory/{team}?season=` | Team trajectory |
| `GET /v1/trajectory/{team}/teammate?season=` | Teammate trajectory |

## Render Deployment

### render.yaml (applied)
```yaml
envVars:
  - key: PYTHONPATH
    value: .
  - key: EPA_DB_PATH
    value: /opt/render/project/cache/epa_data.db
disk:
  name: cache
  mountPath: /opt/render/project/cache
  sizeGB: 1
```

**Important**: `EPA_DB_PATH` env var must be set for the persistent disk mount to work. Without it, the DB goes to ephemeral storage and data is lost on redeploy.

### Known Render Issues
- Pipeline runs sequentially for all 7 seasons (~10+ minutes total). Render health checks may restart during long pipelines.
- GraphQL cache (`cache/ftcscout/`) is NOT on the persistent disk — lost on each deploy.
- `WEB_CONCURRENCY=1` is set automatically by Render.

## Koyeb Deployment

`koyeb.yaml` exists but has **no persistent disk** and **no `EPA_DB_PATH`** set. Data will not persist across redeploys. To fix:
```yaml
env:
  - key: EPA_DB_PATH
    value: /path/to/persistent/epa_data.db
```

## Known Issues (Not Fixed)

1. **100-event GraphQL limit** (`read_ftcscout.py:11`) — seasons with >100 events silently miss data
2. **No chronological ordering** — events processed by code name, not date
3. **No cache invalidation** — `cache/ftcscout/*.p` never refreshed
4. **Double data fetch** — calibration fetches all matches, then pipeline fetches again
5. **Skew only tracked for total dimension** — auto/teleop/endgame skew stays 0
6. **`is_elim` always 0 in `/v1/matches` list** (individual match list, not detail)
7. **50-team cap on unfiltered match listing** (`match.py:29`)
8. **`pyproject.toml` `pip install -e .` broken** — wrong package `include` pattern

## Client Package

Located at `scoutkick-python/`. Pure stdlib, zero deps:
```python
from scoutkick_api import ScoutKick
sk = ScoutKick(base_url="http://127.0.0.1:8000")
sk.get_team(26914)
sk.predict(red=[26914, 32736], blue=[23400, 24599])
```
