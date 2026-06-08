import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np


class SQLiteStorage:
    def __init__(self, db_path: str, season_id: str):
        self.db_path = Path(db_path)
        self.season_id = season_id
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_tables(self):
        conn = self._connect()
        # Migration: recreate team_matches if it has old PK (missing event_code)
        old_schema = conn.execute(
            "SELECT sql FROM sqlite_master WHERE tbl_name='team_matches' AND sql NOT LIKE '%event_code%match_id%'"
        ).fetchone()
        if old_schema:
            conn.execute("DROP TABLE IF EXISTS team_matches")

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS team_seasons (
                team      INTEGER NOT NULL,
                season    TEXT    NOT NULL,
                mean_json TEXT    NOT NULL,
                var_json  TEXT    NOT NULL,
                skew      REAL    NOT NULL DEFAULT 0.0,
                n         REAL    NOT NULL DEFAULT 1.0,
                count     INTEGER NOT NULL DEFAULT 0,
                norm_epa  REAL    DEFAULT NULL,
                updated_at TEXT   NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (team, season)
            );

            CREATE TABLE IF NOT EXISTS team_events (
                team       INTEGER NOT NULL,
                season     TEXT    NOT NULL,
                event_code TEXT    NOT NULL,
                event_type TEXT    DEFAULT NULL,
                epa_start  REAL    DEFAULT NULL,
                epa_max    REAL    DEFAULT NULL,
                epa_pre_elim REAL  DEFAULT NULL,
                epa_mean   REAL    DEFAULT NULL,
                mean_json  TEXT    DEFAULT NULL,
                var_json   TEXT    DEFAULT NULL,
                skew       REAL    DEFAULT NULL,
                n          REAL    DEFAULT NULL,
                count      INTEGER DEFAULT NULL,
                norm_epa   REAL    DEFAULT NULL,
                updated_at TEXT    NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (team, season, event_code)
            );

            CREATE TABLE IF NOT EXISTS team_matches (
                team         INTEGER NOT NULL,
                season       TEXT    NOT NULL,
                event_code   TEXT    NOT NULL,
                match_id     TEXT    NOT NULL,
                epa_pre      REAL    DEFAULT NULL,
                epa_post     REAL    DEFAULT NULL,
                mean_json_pre TEXT   DEFAULT NULL,
                win_prob     REAL    DEFAULT NULL,
                is_elim      INTEGER NOT NULL DEFAULT 0,
                processed_at TEXT    NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (team, season, event_code, match_id)
            );

            CREATE TABLE IF NOT EXISTS seasons (
                season            TEXT    PRIMARY KEY NOT NULL,
                score_mean        REAL    DEFAULT NULL,
                score_sd          REAL    DEFAULT NULL,
                component_means_json TEXT DEFAULT NULL,
                num_matches       INTEGER DEFAULT 0,
                num_teams         INTEGER DEFAULT 0,
                updated_at        TEXT    NOT NULL DEFAULT (datetime('now'))
            );
        """)
        conn.commit()
        conn.close()

    # ── team_seasons ──────────────────────────────────────────────

    def save_team(self, team: int, mean: np.ndarray, var: np.ndarray,
                  skew: float, n: float, count: int, norm_epa: Optional[float] = None):
        conn = self._connect()
        conn.execute("""
            INSERT INTO team_seasons (team, season, mean_json, var_json,
                                      skew, n, count, norm_epa, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(team, season) DO UPDATE SET
                mean_json  = excluded.mean_json,
                var_json   = excluded.var_json,
                skew       = excluded.skew,
                n          = excluded.n,
                count      = excluded.count,
                norm_epa   = excluded.norm_epa,
                updated_at = datetime('now')
        """, (
            team, self.season_id,
            json.dumps(mean.tolist()),
            json.dumps(var.tolist()),
            skew, n, count, norm_epa,
        ))
        conn.commit()
        conn.close()

    def load_team(self, team: int) -> Optional[Dict]:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM team_seasons WHERE team = ? AND season = ?",
            (team, self.season_id),
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return self._row_to_team(row)

    def load_all_teams(self) -> Dict[int, Dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM team_seasons WHERE season = ?",
            (self.season_id,),
        ).fetchall()
        conn.close()
        return {r["team"]: self._row_to_team(r) for r in rows}

    def load_season_teams(self, season_id: str) -> Dict[int, Dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM team_seasons WHERE season = ?",
            (season_id,),
        ).fetchall()
        conn.close()
        return {r["team"]: self._row_to_team(r) for r in rows}

    def load_team_cross_season(self, team: int) -> List[Dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM team_seasons WHERE team = ? ORDER BY season",
            (team,),
        ).fetchall()
        conn.close()
        return [self._row_to_team(r) for r in rows]

    def save_all_teams(self, epas: Dict[int, "SkewNormal"], counts: Dict[int, int],
                       norm_epas: Optional[Dict[int, float]] = None):
        if norm_epas is None:
            norm_epas = {}
        for team, sn in epas.items():
            self.save_team(
                team, sn.mean, sn.var, sn.skew, sn.n,
                counts.get(team, 0), norm_epas.get(team),
            )

    def delete_season(self):
        conn = self._connect()
        conn.execute("DELETE FROM team_seasons WHERE season = ?", (self.season_id,))
        conn.execute("DELETE FROM team_events WHERE season = ?", (self.season_id,))
        conn.execute("DELETE FROM team_matches WHERE season = ?", (self.season_id,))
        conn.execute("DELETE FROM seasons WHERE season = ?", (self.season_id,))
        conn.commit()
        conn.close()

    @staticmethod
    def get_prior_seasons(current_season: str, look_back: int = 4) -> List[str]:
        try:
            curr = int(current_season)
        except ValueError:
            return []
        return [str(curr - i) for i in range(1, look_back + 1)]

    @staticmethod
    def _row_to_team(row) -> Dict:
        return {
            "mean": np.array(json.loads(row["mean_json"])),
            "var": np.array(json.loads(row["var_json"])),
            "skew": row["skew"],
            "n": row["n"],
            "count": row["count"],
            "norm_epa": row["norm_epa"],
        }

    # ── team_events ───────────────────────────────────────────────

    def save_team_event(self, team: int, event_code: str,
                        epa_start: Optional[float] = None,
                        epa_max: Optional[float] = None,
                        epa_pre_elim: Optional[float] = None,
                        epa_mean: Optional[float] = None,
                        mean: Optional[np.ndarray] = None,
                        var: Optional[np.ndarray] = None,
                        skew: Optional[float] = None,
                        n: Optional[float] = None,
                        count: Optional[int] = None,
                        norm_epa: Optional[float] = None,
                        event_type: Optional[str] = None):
        conn = self._connect()
        conn.execute("""
            INSERT INTO team_events (team, season, event_code, event_type,
                epa_start, epa_max, epa_pre_elim, epa_mean,
                mean_json, var_json, skew, n, count, norm_epa, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(team, season, event_code) DO UPDATE SET
                event_type = COALESCE(excluded.event_type, team_events.event_type),
                epa_start  = COALESCE(excluded.epa_start, team_events.epa_start),
                epa_max    = COALESCE(excluded.epa_max, team_events.epa_max),
                epa_pre_elim = COALESCE(excluded.epa_pre_elim, team_events.epa_pre_elim),
                epa_mean   = COALESCE(excluded.epa_mean, team_events.epa_mean),
                mean_json  = COALESCE(excluded.mean_json, team_events.mean_json),
                var_json   = COALESCE(excluded.var_json, team_events.var_json),
                skew       = COALESCE(excluded.skew, team_events.skew),
                n          = COALESCE(excluded.n, team_events.n),
                count      = COALESCE(excluded.count, team_events.count),
                norm_epa   = COALESCE(excluded.norm_epa, team_events.norm_epa),
                updated_at = datetime('now')
        """, (
            team, self.season_id, event_code, event_type,
            epa_start, epa_max, epa_pre_elim, epa_mean,
            json.dumps(mean.tolist()) if mean is not None else None,
            json.dumps(var.tolist()) if var is not None else None,
            skew, n, count, norm_epa,
        ))
        conn.commit()
        conn.close()

    def load_team_events(self) -> List[Dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM team_events WHERE season = ? ORDER BY team, event_code",
            (self.season_id,),
        ).fetchall()
        conn.close()
        return [self._row_to_event(r) for r in rows]

    def load_team_event(self, team: int, event_code: str) -> Optional[Dict]:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM team_events WHERE team = ? AND season = ? AND event_code = ?",
            (team, self.season_id, event_code),
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return self._row_to_event(row)

    def _row_to_event(self, row) -> Dict:
        r = dict(row)
        if r["mean_json"]:
            r["mean"] = np.array(json.loads(r["mean_json"]))
        else:
            r["mean"] = None
        if r["var_json"]:
            r["var"] = np.array(json.loads(r["var_json"]))
        else:
            r["var"] = None
        r.pop("mean_json", None)
        r.pop("var_json", None)
        return r

    # ── team_matches ──────────────────────────────────────────────

    def save_team_match(self, team: int, event_code: str, match_id: str,
                        epa_pre: Optional[float] = None,
                        epa_post: Optional[float] = None,
                        mean_json_pre: Optional[str] = None,
                        win_prob: Optional[float] = None,
                        is_elim: bool = False):
        conn = self._connect()
        conn.execute("""
            INSERT INTO team_matches (team, season, event_code, match_id,
                epa_pre, epa_post, mean_json_pre, win_prob, is_elim, processed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(team, season, event_code, match_id) DO UPDATE SET
                epa_post     = excluded.epa_post,
                mean_json_pre = excluded.mean_json_pre,
                win_prob     = excluded.win_prob,
                is_elim      = excluded.is_elim,
                processed_at = datetime('now')
        """, (
            team, self.season_id, event_code, match_id,
            epa_pre, epa_post, mean_json_pre, win_prob, int(is_elim),
        ))
        conn.commit()
        conn.close()

    def get_processed_match_keys(self) -> Set[tuple]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT DISTINCT event_code, match_id FROM team_matches WHERE season = ?",
            (self.season_id,),
        ).fetchall()
        conn.close()
        return {(r["event_code"], r["match_id"]) for r in rows}

    def load_team_matches(self, team: int) -> List[Dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM team_matches WHERE team = ? AND season = ? ORDER BY processed_at",
            (team, self.season_id),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ── seasons ───────────────────────────────────────────────────

    def save_season_meta(self, score_mean: float, score_sd: float,
                         component_means: np.ndarray,
                         num_matches: int, num_teams: int):
        conn = self._connect()
        conn.execute("""
            INSERT INTO seasons (season, score_mean, score_sd,
                component_means_json, num_matches, num_teams, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(season) DO UPDATE SET
                score_mean           = excluded.score_mean,
                score_sd             = excluded.score_sd,
                component_means_json = excluded.component_means_json,
                num_matches          = excluded.num_matches,
                num_teams            = excluded.num_teams,
                updated_at           = datetime('now')
        """, (
            self.season_id,
            score_mean, score_sd,
            json.dumps(component_means.tolist()),
            num_matches, num_teams,
        ))
        conn.commit()
        conn.close()

    def load_season_meta(self) -> Optional[Dict]:
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM seasons WHERE season = ?",
            (self.season_id,),
        ).fetchone()
        conn.close()
        if row is None:
            return None
        r = dict(row)
        if r["component_means_json"]:
            r["component_means"] = np.array(json.loads(r["component_means_json"]))
        else:
            r["component_means"] = None
        r.pop("component_means_json", None)
        return r
