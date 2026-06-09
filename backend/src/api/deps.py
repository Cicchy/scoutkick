import os
from backend.src.storage.sqlite_storage import SQLiteStorage

DB_PATH = os.environ.get("EPA_DB_PATH") or os.path.join(
    os.getcwd(), "cache", "epa_data.db",
)


def get_storage(season: str = "2025") -> SQLiteStorage:
    return SQLiteStorage(DB_PATH, season)
