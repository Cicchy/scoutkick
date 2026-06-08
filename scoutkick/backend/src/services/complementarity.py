from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from scoutkick.backend.src.services.clustering import DIM_LABELS
from scoutkick.backend.src.storage.sqlite_storage import SQLiteStorage


def _get_playstyle_vec(
    storage: SQLiteStorage,
    team: int,
    season: str,
) -> Optional[np.ndarray]:
    params = storage.load_team(team)
    if params is None:
        return None
    labels = DIM_LABELS.get(season, ["auto", "teleop", "endgame"])
    n_dims = len(labels)
    mean = params["mean"]
    total = float(mean[0])
    if total <= 0:
        return None
    vec = np.array([max(float(mean[idx]), 0) for idx in range(1, n_dims + 1)])
    s = np.sum(vec)
    if s <= 0:
        return None
    return vec / s


def complementarity_score(
    storage: SQLiteStorage,
    team1: int,
    team2: int,
    season: str,
) -> Optional[Dict[str, Any]]:
    v1 = _get_playstyle_vec(storage, team1, season)
    v2 = _get_playstyle_vec(storage, team2, season)
    if v1 is None or v2 is None:
        return None

    labels = DIM_LABELS.get(season, ["auto", "teleop", "endgame"])

    # Coverage: combined strength across dimensions
    combined = np.maximum(v1, v2)
    coverage = float(np.sum(combined))

    # Diversity: how different their playstyles are (0 = identical, 1 = orthogonal)
    dist = float(cosine_distances([v1], [v2])[0][0])

    # Bottleneck: the weakest jointly-covered dimension
    bottleneck_dim = int(np.argmin(combined))
    bottleneck_val = float(combined[bottleneck_dim])

    # Complementarity: coverage × diversity
    # Higher means better collective coverage AND diverse strengths
    score = coverage * (1.0 + dist)

    # Load EPA values for display
    p1 = storage.load_team(team1)
    p2 = storage.load_team(team2)
    epa1 = round(float(p1["mean"][0]), 2) if p1 else None
    epa2 = round(float(p2["mean"][0]), 2) if p2 else None

    return {
        "team1": team1,
        "team2": team2,
        "epa1": epa1,
        "epa2": epa2,
        "complementarity": round(score, 4),
        "coverage": round(coverage, 4),
        "diversity": round(dist, 4),
        "bottleneck_dim": labels[bottleneck_dim] if bottleneck_dim < len(labels) else "unknown",
        "bottleneck_value": round(bottleneck_val, 4),
        "team1_playstyle": {labels[i]: round(float(v1[i]), 4) for i in range(len(labels))},
        "team2_playstyle": {labels[i]: round(float(v2[i]), 4) for i in range(len(labels))},
        "combined_coverage": {labels[i]: round(float(combined[i]), 4) for i in range(len(labels))},
    }


def best_alliance_partners(
    storage: SQLiteStorage,
    team: int,
    season: str,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    v_target = _get_playstyle_vec(storage, team, season)
    if v_target is None:
        return []

    all_teams = storage.load_all_teams()
    labels = DIM_LABELS.get(season, ["auto", "teleop", "endgame"])

    scores: List[Tuple[float, int, float, float]] = []
    for other_num in all_teams:
        if other_num == team:
            continue
        v_other = _get_playstyle_vec(storage, other_num, season)
        if v_other is None:
            continue

        combined = np.maximum(v_target, v_other)
        coverage = float(np.sum(combined))
        dist = float(cosine_distances([v_target], [v_other])[0][0])
        score = coverage * (1.0 + dist)
        other_epa = float(all_teams[other_num]["mean"][0])
        scores.append((score, other_num, coverage, other_epa))

    scores.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, other_num, coverage, other_epa in scores[:top_n]:
        results.append({
            "team": other_num,
            "epa": round(other_epa, 2),
            "complementarity": round(score, 4),
            "coverage": round(coverage, 4),
        })

    return results
