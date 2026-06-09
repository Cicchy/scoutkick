from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from backend.src.api.deps import get_storage

router = APIRouter(tags=["Team"])


@router.get("/v1/teams")
def list_teams(
    season: str = Query("2025", description="Season year"),
    metric: str = Query("norm_epa", description="Sort metric: norm_epa, total, auto, teleop, endgame"),
    ascending: bool = Query(False),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None, description="Filter by team number prefix"),
):
    storage = get_storage(season)
    teams = storage.load_all_teams()

    results = []
    for team, params in teams.items():
        if search is not None and search not in str(team):
            continue
        results.append({
            "team": team,
            "total": float(params["mean"][0]),
            "auto": float(params["mean"][1]),
            "teleop": float(params["mean"][2]),
            "endgame": float(params["mean"][3]),
            "norm_epa": params["norm_epa"],
            "matches": params["count"],
        })

    sort_key_map = {
        "norm_epa": lambda x: x["norm_epa"] if x["norm_epa"] is not None else 0,
        "total": lambda x: x["total"],
        "auto": lambda x: x["auto"],
        "teleop": lambda x: x["teleop"],
        "endgame": lambda x: x["endgame"],
        "team": lambda x: x["team"],
        "matches": lambda x: x["matches"],
    }
    key_fn = sort_key_map.get(metric, sort_key_map["norm_epa"])
    results.sort(key=key_fn, reverse=not ascending)
    sliced = results[offset:offset + limit]
    return {"value": sliced, "count": len(results)}


@router.get("/v1/team/{team}")
def get_team(team: int, season: str = Query("2025")):
    storage = get_storage(season)
    params = storage.load_team(team)
    if params is None:
        raise HTTPException(status_code=404, detail=f"Team {team} not found in season {season}")

    matches = storage.load_team_matches(team)
    team_matches = []
    for m in matches:
        team_matches.append({
            "event_code": m["event_code"],
            "match_id": m["match_id"],
            "epa_pre": m["epa_pre"],
            "epa_post": m["epa_post"],
            "win_prob": m["win_prob"],
            "is_elim": bool(m["is_elim"]),
            "processed_at": m["processed_at"],
        })

    return {
        "team": team,
        "season": season,
        "total": float(params["mean"][0]),
        "auto": float(params["mean"][1]),
        "teleop": float(params["mean"][2]),
        "endgame": float(params["mean"][3]),
        "mean": params["mean"].tolist(),
        "var": params["var"].tolist(),
        "skew": params["skew"],
        "n": params["n"],
        "count": params["count"],
        "norm_epa": params["norm_epa"],
        "team_matches": team_matches,
    }


@router.get("/v1/team/{team}/events")
def get_team_events(
    team: int,
    season: str = Query("2025"),
    event_type: Optional[str] = Query(None, description="Filter by event type (Qualifier, Championship, etc.)"),
    metric: str = Query("epa_mean", description="Sort metric: epa_mean, epa_max, epa_start, count"),
    ascending: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    storage = get_storage(season)
    all_events = storage.load_team_events()
    events = [e for e in all_events if e["team"] == team]

    if event_type is not None:
        events = [e for e in events if e.get("event_type") == event_type]

    for e in events:
        if e.get("mean") is not None:
            e["mean"] = e["mean"].tolist()
        if e.get("var") is not None:
            e["var"] = e["var"].tolist()

    sort_key_map = {
        "epa_mean": lambda x: x.get("epa_mean") or 0,
        "epa_max": lambda x: x.get("epa_max") or 0,
        "epa_start": lambda x: x.get("epa_start") or 0,
        "count": lambda x: x.get("count") or 0,
    }
    key_fn = sort_key_map.get(metric, sort_key_map["epa_mean"])
    events.sort(key=key_fn, reverse=not ascending)
    sliced = events[offset:offset + limit]
    return {"value": sliced, "count": len(events)}


@router.get("/v1/team/{team}/matches")
def get_team_matches(
    team: int,
    season: str = Query("2025"),
    event: Optional[str] = Query(None),
    elim: Optional[bool] = Query(None),
    metric: str = Query("match_id", description="Sort metric: match_id, epa_pre, epa_post, win_prob"),
    ascending: bool = Query(True),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    storage = get_storage(season)
    matches = storage.load_team_matches(team)
    if event is not None:
        matches = [m for m in matches if m["event_code"] == event]
    if elim is not None:
        matches = [m for m in matches if m["is_elim"] == int(elim)]

    for m in matches:
        m["is_elim"] = bool(m["is_elim"])

    sort_key_map = {
        "match_id": lambda x: int(x["match_id"]) if x["match_id"].isdigit() else x["match_id"],
        "epa_pre": lambda x: x["epa_pre"] or 0,
        "epa_post": lambda x: x["epa_post"] or 0,
        "win_prob": lambda x: x["win_prob"] or 0,
    }
    key_fn = sort_key_map.get(metric, sort_key_map["match_id"])
    matches.sort(key=key_fn, reverse=not ascending)
    sliced = matches[offset:offset + limit]
    return {"value": sliced, "count": len(matches)}

