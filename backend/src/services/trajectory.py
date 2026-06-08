from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scoutkick.backend.src.storage.sqlite_storage import SQLiteStorage

MIN_MATCHES = 3
DEFAULT_N_CLUSTERS = 4


def _compute_trajectory_features(
    storage: SQLiteStorage,
    team: int,
    season: str,
) -> Optional[Dict[str, float]]:
    matches = storage.load_team_matches(team)
    if not matches or len(matches) < MIN_MATCHES:
        return None

    epas = np.array([m["epa_post"] for m in matches if m["epa_post"] is not None], dtype=float)
    if len(epas) < MIN_MATCHES:
        return None

    idx = np.arange(len(epas))
    slope, intercept = np.polyfit(idx, epas, 1)

    mid = len(epas) // 2
    first_half = epas[:mid]
    second_half = epas[mid:]
    mid_slope = (np.mean(second_half) - np.mean(first_half)) / max(len(second_half), 1)

    diffs = np.abs(np.diff(epas))
    volatility = float(np.mean(diffs)) if len(diffs) > 0 else 0.0

    return {
        "epa_start": float(epas[0]),
        "epa_end": float(epas[-1]),
        "epa_max": float(np.max(epas)),
        "epa_min": float(np.min(epas)),
        "epa_change": float(epas[-1] - epas[0]),
        "epa_range": float(np.max(epas) - np.min(epas)),
        "epa_mean": float(np.mean(epas)),
        "epa_std": float(np.std(epas)),
        "epa_slope": float(slope),
        "epa_mid_slope": float(mid_slope),
        "epa_volatility": volatility,
        "match_count": len(epas),
    }


def compute_trajectory_clusters(
    storage: SQLiteStorage,
    season: str,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = 42,
) -> Dict[str, Any]:
    all_teams = storage.load_all_teams()

    feat_list: List[Dict] = []
    team_nums: List[int] = []

    for team_num in all_teams:
        feats = _compute_trajectory_features(storage, team_num, season)
        if feats is None:
            continue
        feat_list.append(feats)
        team_nums.append(team_num)

    if len(feat_list) < n_clusters:
        n_clusters = max(2, len(feat_list))

    feature_keys = ["epa_start", "epa_end", "epa_change", "epa_slope", "epa_std", "epa_volatility", "epa_range", "match_count"]
    X = np.array([[f[k] for k in feature_keys] for f in feat_list])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    actual_k = min(n_clusters, len(X_scaled))
    kmeans = KMeans(n_clusters=actual_k, random_state=random_state, n_init="auto")
    assignments = kmeans.fit_predict(X_scaled)

    # Build cluster profiles
    clusters: List[Dict] = []
    for i in range(actual_k):
        mask = assignments == i
        member_indices = np.where(mask)[0]
        member_teams = [team_nums[j] for j in member_indices]
        centroids_unscaled = X[mask].mean(axis=0)
        centroids_scaled = kmeans.cluster_centers_[i]

        cluster_profile = {
            k: round(float(centroids_unscaled[j]), 2)
            for j, k in enumerate(feature_keys)
        }
        cluster_profile_scaled = {
            k: round(float(centroids_scaled[j]), 4)
            for j, k in enumerate(feature_keys)
        }

        label = _auto_trajectory_label(centroids_scaled, feature_keys)
        top_by_epa = sorted(
            member_teams,
            key=lambda t: float(all_teams[t]["mean"][0]),
            reverse=True,
        )[:10]

        clusters.append({
            "id": i,
            "size": int(mask.sum()),
            "label": label,
            "profile": cluster_profile,
            "profile_scaled": cluster_profile_scaled,
            "top_teams": top_by_epa,
        })

    team_results: Dict[str, Dict] = {}
    for i, team_num in enumerate(team_nums):
        cid = int(assignments[i])
        team_results[str(team_num)] = {
            "cluster": cid,
            "features": {k: round(float(feat_list[i][k]), 2) for k in feature_keys},
        }

    return {
        "season": season,
        "n_clusters": actual_k,
        "feature_keys": feature_keys,
        "min_matches": MIN_MATCHES,
        "clusters": clusters,
        "teams": team_results,
    }


def _auto_trajectory_label(centroid_scaled: np.ndarray, keys: List[str]) -> str:
    """Name a trajectory cluster based on its centroid's dominant traits."""
    d = {k: centroid_scaled[i] for i, k in enumerate(keys)}
    slope = d.get("epa_slope", 0)
    std = d.get("epa_std", 0)
    volatility = d.get("epa_volatility", 0)
    change = d.get("epa_change", 0)

    # High volatility
    if volatility > 0.8 or std > 0.8:
        if slope > 0.3:
            return "Volatile Improver"
        elif slope < -0.3:
            return "Volatile Decliner"
        return "Inconsistent"

    # Strong trends
    if slope > 0.8 and change > 0.8:
        return "Late Bloomer"
    if slope < -0.6:
        return "Declining"
    if slope < -0.3:
        return "Fading"

    # Flat trajectory
    if abs(slope) < 0.2 and std < 0.3:
        return "Consistent"
    if abs(slope) < 0.3:
        return "Stable"

    # Mild trends
    if slope > 0.4:
        return "Improving"
    if slope > 0.2:
        return "Gradual Improver"

    return "Mixed"


def get_team_trajectory(
    storage: SQLiteStorage,
    team: int,
    season: str,
) -> Optional[Dict[str, Any]]:
    matches = storage.load_team_matches(team)
    if not matches or len(matches) < MIN_MATCHES:
        return None

    match_data = []
    for m in matches:
        match_data.append({
            "event_code": m["event_code"],
            "match_id": m["match_id"],
            "epa_pre": round(m["epa_pre"], 2) if m["epa_pre"] is not None else None,
            "epa_post": round(m["epa_post"], 2) if m["epa_post"] is not None else None,
            "win_prob": round(m["win_prob"], 4) if m["win_prob"] is not None else None,
            "is_elim": bool(m["is_elim"]),
            "processed_at": m["processed_at"],
        })

    features = _compute_trajectory_features(storage, team, season)
    if features is None:
        return None

    return {
        "team": team,
        "season": season,
        "features": {k: round(v, 2) for k, v in features.items()},
        "matches": match_data,
    }
