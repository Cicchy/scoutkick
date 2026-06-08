from typing import List, Optional, Dict, Any
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scoutkick.backend.src.storage.sqlite_storage import SQLiteStorage

DIM_LABELS: Dict[str, List[str]] = {
    "2025": [
        "auto", "teleop", "endgame",
        "rp1", "rp2", "rp3",
        "auto_classified", "teleop_classified", "teleop_depot",
    ],
}
for sid in ["2019", "2020", "2021", "2022", "2023", "2024"]:
    DIM_LABELS[sid] = ["auto", "teleop", "endgame"]

DEFAULT_N_CLUSTERS = 5


def _extract_playstyle_vectors(teams: Dict[int, Dict], season: str) -> (np.ndarray, List[int]):
    labels = DIM_LABELS.get(season, ["auto", "teleop", "endgame"])
    n_dims = len(labels)
    team_nums: List[int] = []
    vectors: List[np.ndarray] = []

    for team_num, params in teams.items():
        mean = params["mean"]
        total = float(mean[0])
        if total <= 0:
            continue
        vec = np.array([max(float(mean[idx]), 0) for idx in range(1, n_dims + 1)])
        vec_sum = np.sum(vec)
        if vec_sum <= 0:
            continue
        team_nums.append(team_num)
        vectors.append(vec)

    if not vectors:
        return np.array([]), []

    return np.array(vectors), team_nums


def compute_clusters(
    teams: Dict[int, Dict],
    season: str,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = 42,
) -> Dict[str, Any]:
    vectors, team_nums = _extract_playstyle_vectors(teams, season)
    if len(vectors) == 0:
        return {"season": season, "n_clusters": 0, "clusters": [], "teams": {}}

    labels = DIM_LABELS.get(season, ["auto", "teleop", "endgame"])

    # Normalize to playstyle proportions
    normalized = normalize(vectors, norm="l1")

    actual_k = min(n_clusters, len(vectors))
    kmeans = KMeans(n_clusters=actual_k, random_state=random_state, n_init="auto")
    assignments = kmeans.fit_predict(normalized)

    # Build cluster info
    clusters: List[Dict] = []
    for i in range(actual_k):
        mask = assignments == i
        member_indices = np.where(mask)[0]
        member_teams = [team_nums[j] for j in member_indices]
        center = kmeans.cluster_centers_[i]
        total_points = float(np.sum(vectors[mask].mean(axis=0))) if len(member_indices) > 0 else 0.0

        label = _auto_label(center, labels)

        clusters.append({
            "id": i,
            "size": int(mask.sum()),
            "label": label,
            "center": {labels[j]: round(float(center[j]), 4) for j in range(len(labels))},
            "center_raw": {labels[j]: round(float(vectors[member_indices].mean(axis=0)[j]), 2)
                           for j in range(len(labels))} if len(member_indices) > 0 else {},
            "total_points_mean": round(total_points, 2),
            "top_teams": sorted(member_teams, key=lambda t: float(teams[t]["mean"][0]), reverse=True)[:10],
        })

    team_results: Dict[str, Dict] = {}
    for i, team_num in enumerate(team_nums):
        cid = int(assignments[i])
        team_results[str(team_num)] = {
            "cluster": cid,
            "playstyle": {labels[j]: round(float(normalized[i][j]), 4) for j in range(len(labels))},
        }

    return {
        "season": season,
        "n_clusters": actual_k,
        "dimensions": labels,
        "clusters": clusters,
        "teams": team_results,
    }


def _auto_label(center: np.ndarray, labels: List[str]) -> str:
    sorted_idx = np.argsort(center)[::-1]
    primary = labels[sorted_idx[0]] if sorted_idx[0] < len(labels) else ""
    secondary = labels[sorted_idx[1]] if len(sorted_idx) > 1 and sorted_idx[1] < len(labels) else ""
    pval = float(center[sorted_idx[0]])
    sval = float(center[sorted_idx[1]]) if len(sorted_idx) > 1 else 0

    pfmt = primary.replace("_", " ").title()
    sfmt = secondary.replace("_", " ").title() if secondary else ""

    if pval > 0.50:
        return f"{pfmt} Dominant"
    elif pval > 0.40:
        return f"{pfmt}>_{sfmt}" if sval > 0.20 else f"{pfmt} Heavy"
    elif pval > 0.30:
        return f"{pfmt}+{sfmt}" if sval > 0.20 else f"Balanced {pfmt}"
    else:
        return f"{pfmt}+{sfmt}" if all(c > 0.15 for c in center) else "Balanced"


def get_team_cluster_detail(
    storage: SQLiteStorage,
    team_num: int,
    cluster_result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    params = storage.load_team(team_num)
    if params is None:
        return None

    key = str(team_num)
    if key not in cluster_result.get("teams", {}):
        return None

    team_info = cluster_result["teams"][key]
    cid = team_info["cluster"]
    cluster_info = next((c for c in cluster_result["clusters"] if c["id"] == cid), None)

    labels = DIM_LABELS.get(cluster_result["season"], ["auto", "teleop", "endgame"])
    mean = params["mean"]
    raw_vec = [float(mean[idx]) for idx in range(1, len(labels) + 1)]
    total = float(mean[0])

    return {
        "team": team_num,
        "total_epa": round(total, 2),
        "cluster": cid,
        "cluster_label": cluster_info["label"] if cluster_info else None,
        "cluster_size": cluster_info["size"] if cluster_info else None,
        "playstyle": team_info["playstyle"],
        "raw_components": {labels[j]: round(raw_vec[j], 2) for j in range(len(labels))},
        "cluster_center": cluster_info["center"] if cluster_info else None,
    }
