import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.src.api.router import api_router
from backend.src.api.data import router as data_router, pipeline_status
from backend.src.api.deps import get_db_path
from backend.src.storage import create_storage

logger = logging.getLogger(__name__)


def _check_db_has_data() -> bool:
    try:
        storage = create_storage("2025")
        return len(storage.load_all_seasons_meta()) > 0
    except Exception:
        return False


def _get_seasons_in_db() -> list[str]:
    try:
        storage = create_storage("2025")
        return [m["season"] for m in storage.load_all_seasons_meta()]
    except Exception:
        return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_path = get_db_path()
    has_data = _check_db_has_data()
    if not has_data:
        logger.warning(
            "DB is empty (%s). POST /v1/data/run to populate.", db_path,
        )
    else:
        logger.info("DB has existing data — serving. (%s)", db_path)
    yield


app = FastAPI(title="scoutkick EPA API", version="0.1.0", lifespan=lifespan)
app.include_router(api_router)
app.include_router(data_router)


@app.get("/")
def root():
    return {
        "name": "scoutkick",
        "version": "0.1.0",
        "docs": "/docs",
        "pipeline": pipeline_status,
    }


@app.get("/v1/pipeline")
def get_pipeline_status():
    return pipeline_status


@app.get("/v1/health")
def health():
    has_data = _check_db_has_data()
    seasons_in_db = _get_seasons_in_db()
    return {
        "status": "ok" if has_data else "degraded",
        "db_has_data": has_data,
        "seasons": seasons_in_db,
        "pipeline": pipeline_status["state"],
        "pipeline_running": pipeline_status["state"] == "running",
    }
