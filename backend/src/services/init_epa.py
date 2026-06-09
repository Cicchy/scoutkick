from typing import Dict, Optional

import numpy as np
from scipy import stats

from backend.src.core.config import SeasonConfig
from backend.src.core.math import SkewNormal, inv_unit_sigmoid

NORM_MEAN = 1500.0
NORM_SD = 250.0
INIT_PENALTY = 0.2
YEAR_ONE_WEIGHT = 0.7
MEAN_REVERSION = 0.4

INIT_EPA = NORM_MEAN - INIT_PENALTY * NORM_SD  # 1450

NUM_BREAKPOINTS = 101


def _zscore_norm_epa(epas: Dict[int, SkewNormal]) -> Dict[int, float]:
    totals = np.array([sn.mean[0] for sn in epas.values()])
    mean_epa = float(np.mean(totals))
    sd_epa = float(np.std(totals))
    if sd_epa < 1e-10:
        return {team: NORM_MEAN for team in epas}
    result = {}
    for team, sn in epas.items():
        z = (sn.mean[0] - mean_epa) / sd_epa
        result[team] = round(NORM_MEAN + NORM_SD * z, 2)
    return result


def compute_norm_epa(epas: Dict[int, SkewNormal]) -> Dict[int, float]:
    totals = np.array([sn.mean[0] for sn in epas.values()])

    if len(totals) < 5:
        return _zscore_norm_epa(epas)

    try:
        params = stats.exponnorm.fit(totals)
    except Exception:
        return _zscore_norm_epa(epas)

    K, loc, scale = params
    distrib = stats.exponnorm(K, loc, scale)

    breakpoints = np.linspace(0, 1, NUM_BREAKPOINTS)
    quantiles = distrib.ppf(breakpoints)

    result = {}
    for team, sn in epas.items():
        raw = sn.mean[0]
        raw = max(float(quantiles[0]), min(float(quantiles[-1]), raw))
        p = float(np.interp(raw, quantiles, breakpoints))
        p = max(1e-6, min(1 - 1e-6, p))
        norm_epa = NORM_MEAN + NORM_SD * float(stats.norm.ppf(p))
        result[team] = round(norm_epa, 2)

    return result


def get_init_epa(
    config: SeasonConfig,
    component_means: np.ndarray,
    score_sd: float,
    score_mean: float,
    prior_norm_epa_1: Optional[float] = None,
    prior_norm_epa_2: Optional[float] = None,
    mean_reversion: float = MEAN_REVERSION,
) -> SkewNormal:
    num_teams = 2

    norm_epa_1 = prior_norm_epa_1 if prior_norm_epa_1 is not None else INIT_EPA
    norm_epa_2 = prior_norm_epa_2 if prior_norm_epa_2 is not None else INIT_EPA

    prev_norm_epa = YEAR_ONE_WEIGHT * norm_epa_1 + (1 - YEAR_ONE_WEIGHT) * norm_epa_2
    curr_norm_epa = (1 - mean_reversion) * prev_norm_epa + mean_reversion * INIT_EPA

    z_score = (curr_norm_epa - NORM_MEAN) / NORM_SD

    sd_frac = (score_sd / score_mean) if score_mean > 0 else 0.5
    sd = component_means * sd_frac

    mean = component_means.copy()
    if config.rp_indices:
        eps = 1e-6
        for i in config.rp_indices:
            rp_mean = mean[i]
            rp_mean = max(eps, min(1 - eps, rp_mean))
            mean[i] = max(-1.0, inv_unit_sigmoid(rp_mean))

    curr_epa_mean = mean / num_teams + sd * z_score
    curr_epa_var = (sd / num_teams) ** 2

    return SkewNormal(curr_epa_mean, curr_epa_var, 0)
