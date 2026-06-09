from typing import Optional, List
from fastapi import APIRouter, Query, HTTPException
from backend.src.api.deps import get_storage
from backend.src.services.clustering import compute_clusters, get_team_cluster_detail, DEFAULT_N_CLUSTERS
from backend.src.services.complementarity import complementarity_score, best_alliance_partners
from backend.src.services.trajectory import (
    compute_trajectory_clusters,
    get_team_trajectory,
)

router = APIRouter(tags=["Cluster"])


@router.get("/v1/clusters")
def list_clusters(
    season: str = Query("2025", description="Season year"),
    n_clusters: int = Query(DEFAULT_N_CLUSTERS, ge=2, le=20, description="Number of playstyle clusters"),
):
    storage = get_storage(season)
    teams = storage.load_all_teams()
    if not teams:
        raise HTTPException(status_code=404, detail=f"No teams found for season {season}")

    result = compute_clusters(teams, season, n_clusters=n_clusters)
    return result


@router.get("/v1/team/{team}/playstyle")
def get_team_playstyle(
    team: int,
    season: str = Query("2025", description="Season year"),
    n_clusters: int = Query(DEFAULT_N_CLUSTERS, ge=2, le=20),
):
    storage = get_storage(season)
    params = storage.load_team(team)
    if params is None:
        raise HTTPException(status_code=404, detail=f"Team {team} not found in season {season}")

    teams = storage.load_all_teams()
    cluster_result = compute_clusters(teams, season, n_clusters=n_clusters)
    detail = get_team_cluster_detail(storage, team, cluster_result)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Could not classify team {team}")

    return detail


@router.get("/v1/complementarity")
def get_complementarity(
    team1: int = Query(..., description="First team number"),
    team2: int = Query(..., description="Second team number"),
    season: str = Query("2025", description="Season year"),
):
    storage = get_storage(season)
    result = complementarity_score(storage, team1, team2, season)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Could not compute complementarity for {team1}/{team2} in season {season}",
        )
    return result


@router.get("/v1/complementarity/{team}/partners")
def get_alliance_partners(
    team: int,
    season: str = Query("2025", description="Season year"),
    top_n: int = Query(10, ge=1, le=50, description="Number of top partners"),
):
    storage = get_storage(season)
    params = storage.load_team(team)
    if params is None:
        raise HTTPException(status_code=404, detail=f"Team {team} not found in season {season}")

    partners = best_alliance_partners(storage, team, season, top_n=top_n)
    return {"team": team, "season": season, "partners": partners}


@router.get("/v1/trajectory/clusters")
def list_trajectory_clusters(
    season: str = Query("2025", description="Season year"),
    n_clusters: int = Query(4, ge=2, le=10, description="Number of trajectory clusters"),
):
    storage = get_storage(season)
    result = compute_trajectory_clusters(storage, season, n_clusters=n_clusters)
    if not result["clusters"]:
        raise HTTPException(status_code=404, detail=f"No trajectory data available for season {season}")
    return result


@router.get("/v1/team/{team}/trajectory")
def get_team_trajectory_endpoint(
    team: int,
    season: str = Query("2025", description="Season year"),
):
    storage = get_storage(season)
    params = storage.load_team(team)
    if params is None:
        raise HTTPException(status_code=404, detail=f"Team {team} not found in season {season}")

    result = get_team_trajectory(storage, team, season)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Not enough match data for team {team} (need at least 3 matches)",
        )
    return result
