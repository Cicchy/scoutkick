from fastapi import APIRouter
from backend.src.api.season import router as season_router
from backend.src.api.team import router as team_router
from backend.src.api.event import router as event_router
from backend.src.api.match import router as match_router
from backend.src.api.predict import router as predict_router
from backend.src.api.cluster import router as cluster_router

api_router = APIRouter()
api_router.include_router(season_router)
api_router.include_router(team_router)
api_router.include_router(event_router)
api_router.include_router(match_router)
api_router.include_router(predict_router)
api_router.include_router(cluster_router)
