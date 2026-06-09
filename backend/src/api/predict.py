from fastapi import APIRouter, Query, HTTPException
from backend.src.core.config import get_season_config
from backend.src.services.epa_service import EPAEngine
from backend.src.api.deps import get_storage

router = APIRouter(tags=["Predict"])


def _load_engine(season: str) -> EPAEngine:
    config = get_season_config(season)
    engine = EPAEngine(config=config)
    storage = get_storage(season)
    meta = storage.load_season_meta()
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Season {season} not found. Run pipeline first.")
    engine.score_sd = meta["score_sd"]
    teams = storage.load_all_teams()
    for team, params in teams.items():
        engine.set_team_state(
            team, params["mean"], params["var"],
            params["skew"], params["n"], params["count"],
        )
    return engine


@router.get("/v1/predict")
def predict_match(
    red: str = Query(..., description="Comma-separated team numbers, e.g. '26914,32736'"),
    blue: str = Query(..., description="Comma-separated team numbers, e.g. '23400,24599'"),
    season: str = Query("2025"),
):
    red_teams = [int(t.strip()) for t in red.split(",") if t.strip()]
    blue_teams = [int(t.strip()) for t in blue.split(",") if t.strip()]

    if len(red_teams) != 2 or len(blue_teams) != 2:
        raise HTTPException(status_code=400, detail="Must provide exactly 2 teams per alliance")

    engine = _load_engine(season)

    for t in red_teams + blue_teams:
        if t not in engine.epas:
            raise HTTPException(status_code=404, detail=f"Team {t} not found in season {season}")

    team_data = {}
    for t in red_teams + blue_teams:
        sn = engine.get_team(t)
        team_data[t] = {
            "total": float(sn.mean[0]),
            "auto": float(sn.mean[1]),
            "teleop": float(sn.mean[2]),
            "endgame": float(sn.mean[3]),
            "norm_epa": None,
        }

    storage = get_storage(season)
    all_teams = storage.load_all_teams()
    for t in team_data:
        if t in all_teams:
            team_data[t]["norm_epa"] = all_teams[t].get("norm_epa")

    win_prob, red_pred, blue_pred = engine.predict_match(red_teams, blue_teams)

    dims = ["total", "auto", "teleop", "endgame", "rp1", "rp2", "rp3",
            "auto_classified", "teleop_classified", "teleop_depot"]
    red_pred_dict = {dims[i]: float(red_pred[i]) for i in range(len(dims)) if i < len(red_pred)}
    blue_pred_dict = {dims[i]: float(blue_pred[i]) for i in range(len(dims)) if i < len(blue_pred)}

    return {
        "red_teams": [{"team": t, **team_data[t]} for t in red_teams],
        "blue_teams": [{"team": t, **team_data[t]} for t in blue_teams],
        "red_win_prob": round(win_prob, 4),
        "blue_win_prob": round(1 - win_prob, 4),
        "predicted_red": red_pred_dict,
        "predicted_blue": blue_pred_dict,
    }


@router.get("/v1/compare")
def compare_teams(
    teams: str = Query(..., description="Comma-separated team numbers, e.g. '26914,32736,23400'"),
    season: str = Query("2025"),
):
    team_nums = [int(t.strip()) for t in teams.split(",") if t.strip()]
    if len(team_nums) < 2:
        raise HTTPException(status_code=400, detail="Must provide at least 2 teams")

    engine = _load_engine(season)

    results = []
    for t in team_nums:
        if t not in engine.epas:
            raise HTTPException(status_code=404, detail=f"Team {t} not found in season {season}")
        sn = engine.get_team(t)
        results.append({
            "team": t,
            "total": float(sn.mean[0]),
            "auto": float(sn.mean[1]),
            "teleop": float(sn.mean[2]),
            "endgame": float(sn.mean[3]),
            "variance": float(sn.var[0]),
            "skew": sn.skew,
            "matches": engine.counts.get(t, 0),
        })

    return {"season": season, "teams": results}
