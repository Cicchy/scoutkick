from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from scoutkick.backend.src.api.deps import get_storage

router = APIRouter(tags=["Match"])


@router.get("/v1/matches")
def list_matches(
    season: str = Query("2025"),
    event: Optional[str] = Query(None),
    elim: Optional[bool] = Query(None),
    team: Optional[int] = Query(None),
    metric: str = Query("processed_at", description="Sort metric: processed_at, match_id, epa_pre, epa_post, win_prob"),
    ascending: bool = Query(False),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    storage = get_storage(season)
    all_events = storage.load_team_events()
    all_teams = {e["team"] for e in all_events}

    if team is not None:
        if team not in all_teams:
            raise HTTPException(status_code=404, detail=f"Team {team} not found in season {season}")
        match_list = storage.load_team_matches(team)
    else:
        match_list = []
        for t in list(all_teams)[:50]:
            match_list.extend(storage.load_team_matches(t))

    if event is not None:
        match_list = [m for m in match_list if m["event_code"] == event]
    if elim is not None:
        match_list = [m for m in match_list if m["is_elim"] == int(elim)]

    for m in match_list:
        m["is_elim"] = bool(m["is_elim"])

    sort_key_map = {
        "processed_at": lambda x: x["processed_at"] if "processed_at" in x else "",
        "match_id": lambda x: int(x["match_id"]) if x["match_id"].isdigit() else x["match_id"],
        "epa_pre": lambda x: x["epa_pre"] or 0,
        "epa_post": lambda x: x["epa_post"] or 0,
        "win_prob": lambda x: x["win_prob"] or 0,
    }
    key_fn = sort_key_map.get(metric, sort_key_map["processed_at"])
    match_list.sort(key=key_fn, reverse=not ascending)
    sliced = match_list[offset:offset + limit]
    return {"value": sliced, "count": len(match_list)}


@router.get("/v1/match/{event_code}/{match_id}")
def get_match(
    event_code: str,
    match_id: str,
    season: str = Query("2025"),
):
    storage = get_storage(season)
    all_events = storage.load_team_events()
    teams_in_event = {e["team"] for e in all_events if e["event_code"] == event_code}

    if not teams_in_event:
        raise HTTPException(status_code=404, detail=f"Event {event_code} not found")

    teams = []
    for team in teams_in_event:
        tm = storage.load_team_matches(team)
        for m in tm:
            if m["event_code"] == event_code and m["match_id"] == match_id:
                teams.append({
                    "team": m["team"],
                    "epa_pre": m["epa_pre"],
                    "epa_post": m["epa_post"],
                    "win_prob": m["win_prob"],
                })
                break

    if not teams:
        raise HTTPException(status_code=404, detail=f"Match {match_id} at {event_code} not found")

    return {
        "event_code": event_code,
        "match_id": match_id,
        "season": season,
        "is_elim": bool(teams[0].get("is_elim", 0)) if teams else False,
        "teams": teams,
    }
