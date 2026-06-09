import os
from backend.src.storage.sqlite_storage import SQLiteStorage

_API_DIR = os.path.dirname(os.path.abspath(__file__))  # .../scoutkick/backend/src/api
DB_PATH = os.path.join(_API_DIR, "..", "..", "..", "..", "cache", "epa_data.db")


def get_storage(season: str = "2025") -> SQLiteStorage:
    return SQLiteStorage(DB_PATH, season)
