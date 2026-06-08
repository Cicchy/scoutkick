from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from scoutkick.backend.src.api.deps import get_storage

router = APIRouter(tags=["Event"])


@router.get("/v1/events")
def list_events(
    season: str = Query("2025"),
    event_type: Optional[str] = Query(None, description="Filter by event type (Qualifier, Championship, etc.)"),
    metric: str = Query("epa_max", description="Sort metric: epa_max, epa_mean, teams, event_code"),
    ascending: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    storage = get_storage(season)
    events_data = storage.load_team_events()

    event_map = {}
    for e in events_data:
        if event_type is not None and e.get("event_type") != event_type:
            continue
        ec = e["event_code"]
        if ec not in event_map:
            event_map[ec] = {
                "event_code": ec,
                "event_type": e.get("event_type"),
                "team_count": 0,
                "epa_max": float("-inf"),
                "epa_mean_sum": 0.0,
                "epa_mean_count": 0,
            }
        em = event_map[ec]
        em["team_count"] += 1
        if e.get("epa_max") is not None:
            em["epa_max"] = max(em["epa_max"], e["epa_max"])
        if e.get("epa_mean") is not None:
            em["epa_mean_sum"] += e["epa_mean"]
            em["epa_mean_count"] += 1

    results = []
    for ec, info in event_map.items():
        results.append({
            "event_code": ec,
            "event_type": info["event_type"],
            "team_count": info["team_count"],
            "epa_max": info["epa_max"] if info["epa_max"] != float("-inf") else None,
            "epa_mean": info["epa_mean_sum"] / info["epa_mean_count"] if info["epa_mean_count"] > 0 else None,
        })

    sort_key_map = {
        "epa_max": lambda x: x["epa_max"] or 0,
        "epa_mean": lambda x: x["epa_mean"] or 0,
        "teams": lambda x: x["team_count"],
        "event_code": lambda x: x["event_code"],
    }
    key_fn = sort_key_map.get(metric, sort_key_map["epa_max"])
    results.sort(key=key_fn, reverse=not ascending)
    sliced = results[offset:offset + limit]
    return {"value": sliced, "count": len(results)}


@router.get("/v1/event/{event_code}")
def get_event(event_code: str, season: str = Query("2025")):
    storage = get_storage(season)
    all_events = storage.load_team_events()
    event_teams = [e for e in all_events if e["event_code"] == event_code]

    if not event_teams:
        raise HTTPException(status_code=404, detail=f"Event {event_code} not found in season {season}")

    teams = []
    for e in event_teams:
        teams.append({
            "team": e["team"],
            "epa_start": e.get("epa_start"),
            "epa_max": e.get("epa_max"),
            "epa_mean": e.get("epa_mean"),
            "epa_pre_elim": e.get("epa_pre_elim"),
            "count": e.get("count"),
            "norm_epa": e.get("norm_epa"),
        })
    teams.sort(key=lambda x: x["epa_start"] or 0, reverse=True)

    return {
        "event_code": event_code,
        "event_type": event_teams[0].get("event_type"),
        "season": season,
        "team_count": len(teams),
        "teams": teams,
    }


@router.get("/v1/event/{event_code}/matches")
def get_event_matches(
    event_code: str,
    season: str = Query("2025"),
    elim: Optional[bool] = Query(None),
    metric: str = Query("match_id", description="Sort metric: match_id"),
    ascending: bool = Query(True),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    storage = get_storage(season)
    all_events = storage.load_team_events()
    teams_in_event = {e["team"] for e in all_events if e["event_code"] == event_code}

    if not teams_in_event:
        raise HTTPException(status_code=404, detail=f"Event {event_code} not found in season {season}")

    all_matches = []
    for team in teams_in_event:
        all_matches.extend(storage.load_team_matches(team))

    match_map = {}
    for m in all_matches:
        if m["event_code"] != event_code:
            continue
        key = (m["event_code"], m["match_id"])
        if key not in match_map:
            match_map[key] = {
                "event_code": m["event_code"],
                "match_id": m["match_id"],
                "is_elim": bool(m["is_elim"]),
                "teams": [],
            }
        match_map[key]["teams"].append({
            "team": m["team"],
            "epa_pre": m["epa_pre"],
            "epa_post": m["epa_post"],
            "win_prob": m["win_prob"],
        })

    results = list(match_map.values())
    if elim is not None:
        results = [r for r in results if r["is_elim"] == elim]

    results.sort(key=lambda x: int(x["match_id"]) if x["match_id"].isdigit() else x["match_id"],
                 reverse=not ascending)
    sliced = results[offset:offset + limit]
    return {"value": sliced, "count": len(results)}
