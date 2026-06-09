import json
from collections import defaultdict
from typing import Dict, List, Optional, Set

import numpy as np

from backend.src.core.config import get_season_config
from backend.src.data.read_ftcscout import get_matches
from backend.src.data.cleaner import CleanerRegistry
from backend.src.services.epa_service import EPAEngine
from backend.src.services.calibrate import calibrate_score_sd, calibrate_component_means
from backend.src.services.init_epa import get_init_epa, compute_norm_epa
from backend.src.storage.sqlite_storage import SQLiteStorage


class EPAPipeline:
    def __init__(self, season_id: str, db_path: str = "cache/epa_data.db",
                 calibrate: bool = True):
        self.season_id = season_id
        self.config = get_season_config(season_id)
        self.cleaner = CleanerRegistry.get_cleaner(season_id)
        self.engine = EPAEngine(config=self.config)
        self.storage = SQLiteStorage(db_path, season_id)
        self.do_calibrate = calibrate

    def _unique_teams(self, matches: List[Dict]) -> Set[int]:
        teams: Set[int] = set()
        for m in matches:
            teams.update(m.get("red_teams", []))
            teams.update(m.get("blue_teams", []))
        return teams

    def _init_teams(self, teams: Set[int], component_means: np.ndarray,
                    score_sd: float, score_mean: float):
        prior_seasons = self.storage.get_prior_seasons(self.season_id, look_back=4)
        prior_data: Dict[int, Dict[int, Optional[float]]] = {}
        for season in prior_seasons:
            season_teams = self.storage.load_season_teams(season)
            for team, params in season_teams.items():
                prior_data.setdefault(team, {})[int(season)] = params.get("norm_epa")

        initialized = 0
        rookies = 0
        for team in sorted(teams):
            if team in self.engine.epas:
                continue

            past: List[Optional[float]] = []
            for s in prior_seasons:
                s_int = int(s)
                if team in prior_data and s_int in prior_data[team]:
                    past.append(prior_data[team][s_int])
                else:
                    past.append(None)

            prior_1 = past[0] if len(past) > 0 else None
            prior_2 = past[1] if len(past) > 1 else None

            sn = get_init_epa(
                self.config, component_means, score_sd, score_mean,
                prior_norm_epa_1=prior_1, prior_norm_epa_2=prior_2,
            )
            self.engine.set_team_state(team, sn.mean, sn.var, sn.skew, sn.n, 0)
            if prior_1 is not None:
                initialized += 1
            else:
                rookies += 1

        print(f"Initialized {initialized} returning + {rookies} rookie teams from {len(teams)} total.")

    def run(self):
        print(f"--- Starting EPA Pipeline for {self.season_id} ---")

        print("Fetching matches from FTCscout...")
        matches = get_matches(self.cleaner)
        print(f"Found {len(matches)} matches.")

        if not matches:
            print("No matches found. Exiting.")
            return None

        processed_keys = self.storage.get_processed_match_keys()
        new_matches = [m for m in matches if (m["event"], str(m["match_id"])) not in processed_keys]
        if len(new_matches) < len(matches):
            print(f"Skipping {len(matches) - len(new_matches)} already-processed matches ({len(new_matches)} new).")
        matches = new_matches

        if not matches:
            print("No new matches to process.")
            existing = self.storage.load_all_teams()
            if existing:
                for team, params in existing.items():
                    self.engine.set_team_state(
                        team, params["mean"], params["var"],
                        params["skew"], params["n"], params["count"],
                    )
            return self.engine

        if self.do_calibrate:
            score_sd = calibrate_score_sd(self.cleaner, self.season_id)
            self.engine.score_sd = score_sd
            component_means = calibrate_component_means(self.cleaner, self.season_id)
            score_mean = float(component_means[0]) if len(component_means) > 0 else 0.0
            print(f"Calibrated: score_sd={score_sd:.2f}, score_mean={score_mean:.2f}")
        else:
            component_means = np.zeros(len(self.config.dimensions))
            component_means[0] = self.config.default_mean_total * 2
            score_mean = component_means[0]
            score_sd = self.engine.score_sd

        existing = self.storage.load_all_teams()
        if existing:
            print(f"Loaded {len(existing)} teams from storage (resuming).")
            for team, params in existing.items():
                self.engine.set_team_state(
                    team, params["mean"], params["var"],
                    params["skew"], params["n"], params["count"],
                )

        all_teams = self._unique_teams(matches)
        self._init_teams(all_teams, component_means, score_sd, score_mean)

        events: Dict[str, List[Dict]] = defaultdict(list)
        for m in matches:
            events[m["event"]].append(m)

        # Sort matches within each event by match_id
        for ec in events:
            events[ec].sort(key=lambda x: str(x["match_id"]))

        # team -> {event_code -> tracker}
        event_tracker: Dict[int, Dict[str, Dict]] = defaultdict(dict)
        total = sum(len(v) for v in events.values())
        done = 0

        for event_code in sorted(events.keys()):
            event_matches = events[event_code]
            event_type = event_matches[0].get("event_type") if event_matches else None
            print(f"  Event {event_code} ({event_type}): {len(event_matches)} matches")

            # Initialize event tracking for all teams in this event
            for m in event_matches:
                for team in m["red_teams"] + m["blue_teams"]:
                    if event_code not in event_tracker[team]:
                        current_epa = self.engine.get_team(team).mean[0]
                        event_tracker[team][event_code] = {
                            "epa_start": current_epa,
                            "epa_max": current_epa,
                            "epa_sum": 0.0,
                            "match_count": 0,
                        }

            for m in event_matches:
                match_id = str(m["match_id"])
                red_teams = m["red_teams"]
                blue_teams = m["blue_teams"]

                # Pre-match: record EPA for each team
                pre_epas: Dict[int, float] = {}
                pre_means: Dict[int, str] = {}
                for team in red_teams + blue_teams:
                    sn = self.engine.get_team(team)
                    pre_epas[team] = float(sn.mean[0])
                    pre_means[team] = json.dumps(sn.mean.tolist())

                red_raw = self.cleaner.aggregate(m["red_scores"])
                blue_raw = self.cleaner.aggregate(m["blue_scores"])
                red_actual = self.cleaner.clean(red_raw)
                blue_actual = self.cleaner.clean(blue_raw)
                red_weights = self.cleaner.get_attribution_weights(m["red_scores"], red_teams)
                blue_weights = self.cleaner.get_attribution_weights(m["blue_scores"], blue_teams)
                win_prob, red_pred, blue_pred = self.engine.predict_match(red_teams, blue_teams)
                attributions = self.engine.attribute_match(
                    red_teams, blue_teams,
                    red_actual, blue_actual,
                    red_pred, blue_pred,
                    red_weights=red_weights,
                    blue_weights=blue_weights,
                )

                tl = m.get("tournament_level")
                is_elim = tl is not None and tl != "Quals"

                for team_num, attrib in attributions.items():
                    self.engine.update_team(team_num, attrib, is_elim=is_elim)

                # Post-match: save team_match, update tracker
                for team in red_teams + blue_teams:
                    epa_post = float(self.engine.get_team(team).mean[0])
                    self.storage.save_team_match(
                        team, event_code, match_id,
                        epa_pre=pre_epas[team],
                        epa_post=epa_post,
                        mean_json_pre=pre_means[team],
                        win_prob=win_prob if team in red_teams else 1 - win_prob,
                        is_elim=is_elim,
                    )
                    trk = event_tracker[team][event_code]
                    trk["epa_max"] = max(trk["epa_max"], epa_post)
                    trk["epa_sum"] += (pre_epas[team] + epa_post) / 2
                    trk["match_count"] += 1

                done += 1
                if done % 50 == 0:
                    print(f"  Processed {done}/{total} matches...")

            # Save team_event rows for this event
            for team, trk in event_tracker.items():
                if event_code not in trk:
                    continue
                t = trk[event_code]
                sn = self.engine.get_team(team)
                epa_mean = t["epa_sum"] / t["match_count"] if t["match_count"] > 0 else None
                self.storage.save_team_event(
                    team, event_code,
                    event_type=event_type,
                    epa_start=t["epa_start"],
                    epa_max=t["epa_max"],
                    epa_mean=epa_mean,
                    mean=sn.mean, var=sn.var,
                    skew=sn.skew, n=sn.n, count=self.engine.counts.get(team, 0),
                )

        score_mean = float(component_means[0]) if len(component_means) > 0 else 0.0
        self.storage.save_season_meta(
            score_mean=score_mean,
            score_sd=self.engine.score_sd,
            component_means=component_means,
            num_matches=total,
            num_teams=len(self.engine.epas),
        )

        norm_epas = compute_norm_epa(self.engine.epas)
        self.storage.save_all_teams(self.engine.epas, self.engine.counts, norm_epas)
        print(f"Saved {len(self.engine.epas)} teams to storage with norm_epa.")

        print(f"--- Pipeline Complete. Trained {len(self.engine.epas)} teams across {len(events)} events ({total} matches). ---")
        return self.engine


if __name__ == "__main__":
    try:
        pipeline = EPAPipeline("2025")
        engine = pipeline.run()
        if engine:
            sample_team = list(engine.epas.keys())[0]
            print(f"Sample Team {sample_team} Mean Total: {engine.get_team(sample_team).mean[0]:.2f}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Pipeline failed: {e}")
