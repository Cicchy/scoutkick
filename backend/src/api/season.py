import numpy as np
from fastapi import APIRouter, HTTPException
from backend.src.api.deps import get_storage

router = APIRouter(tags=["Season"])


def _serialize_meta(meta: dict) -> dict:
    r = dict(meta)
    for k, v in r.items():
        if isinstance(v, np.ndarray):
            r[k] = v.tolist()
        elif isinstance(v, (np.integer,)):
            r[k] = int(v)
        elif isinstance(v, (np.floating,)):
            r[k] = float(v)
    return r


@router.get("/v1/seasons")
def list_seasons():
    storage = get_storage()
    meta = storage.load_season_meta()
    if meta is None:
        return []
    return [_serialize_meta(meta)]


@router.get("/v1/season/{season}")
def get_season(season: str):
    storage = get_storage(season)
    meta = storage.load_season_meta()
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Season {season} not found")
    return _serialize_meta(meta)
