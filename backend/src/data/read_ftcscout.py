from typing import Any, Dict, List
from scoutkick.backend.src.data.ftcscout_api import get_ftcscout
from scoutkick.backend.src.data.cleaner import BaseCleaner


def get_matches(cleaner: BaseCleaner, cache: bool = True) -> List[Dict[str, Any]]:
    score_fragment = cleaner.get_graphql_fragment()

    query = f"""
    query {{
      eventsSearch(season: {cleaner.SEASON_ID}, limit: 100) {{
        code
        type
        matches {{
          id
          hasBeenPlayed
          tournamentLevel
          teams {{
            teamNumber
            alliance
            station
          }}
          scores {{
            {score_fragment}
          }}
        }}
      }}
    }}
    """

    data = get_ftcscout(query, variables=None, cache=cache)

    if not data or "eventsSearch" not in data:
        return []

    all_matches = []
    for event in data["eventsSearch"]:
        event_code = event["code"]
        event_type = event.get("type")
        matches_in_event = event.get("matches", [])
        if not matches_in_event:
            continue

        for m in matches_in_event:
            if not m.get("hasBeenPlayed"):
                continue

            red_teams = []
            blue_teams = []
            for t in m.get("teams", []):
                if not t:
                    continue
                t_num = t.get("teamNumber")
                alliance = t.get("alliance")
                if alliance == "Red":
                    red_teams.append(t_num)
                elif alliance == "Blue":
                    blue_teams.append(t_num)

            scores_obj = m.get("scores") or {}
            red_scores = scores_obj.get("red") or {}
            blue_scores = scores_obj.get("blue") or {}

            # Remote events (2020/2021) have flat alliance-based scores
            if (not red_scores or not blue_scores) and "alliance" in scores_obj:
                raw = dict(scores_obj)
                alliance_name = raw.pop("alliance", None)
                if alliance_name == "Red":
                    red_scores = raw
                elif alliance_name == "Blue":
                    blue_scores = raw

            # Skip remote/solo matches where only one alliance has teams or scores
            if not red_teams or not blue_teams or not red_scores or not blue_scores:
                continue

            match_info = {
                "match_id": m["id"],
                "event": event_code,
                "event_type": event_type,
                "tournament_level": m.get("tournamentLevel"),
                "red_teams": red_teams,
                "blue_teams": blue_teams,
                "red_scores": red_scores,
                "blue_scores": blue_scores,
            }
            all_matches.append(match_info)

    return all_matches
