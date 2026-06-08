import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scoutkick.backend.src.core.math import SkewNormal, unit_sigmoid
from scoutkick.backend.src.core.config import SeasonConfig

class EPAEngine:
    def __init__(self, config: SeasonConfig):
        self.config = config
        self.score_sd = config.score_sd
        self.k = -5 / 8
        self.num_teams = 2
        self.epas: Dict[int, SkewNormal] = {}
        self.counts: Dict[int, int] = {}

    def get_team(self, team_num: int) -> SkewNormal:
        if team_num not in self.epas:
            init_mean = np.zeros(len(self.config.dimensions))
            init_mean[0] = self.config.default_mean_total
            init_var = np.ones(len(self.config.dimensions)) * self.config.default_variance
            self.epas[team_num] = SkewNormal(init_mean, init_var)
            self.counts[team_num] = 0
        return self.epas[team_num]

    def set_team_state(self, team_num: int, mean: np.ndarray, var: np.ndarray, skew: float, n: float, count: int):
        sn = SkewNormal(mean.copy(), var.copy())
        sn.skew = skew
        sn.n = n
        self.epas[team_num] = sn
        self.counts[team_num] = count

    def predict_match(self, red_teams: List[int], blue_teams: List[int]) -> Tuple[float, np.ndarray, np.ndarray]:
        red_mean = np.array([self.get_team(t).mean for t in red_teams]).sum(axis=0)
        blue_mean = np.array([self.get_team(t).mean for t in blue_teams]).sum(axis=0)

        for i in self.config.rp_indices:
            red_mean[i] = unit_sigmoid(red_mean[i])
            blue_mean[i] = unit_sigmoid(blue_mean[i])

        norm_diff = (red_mean[0] - blue_mean[0]) / self.score_sd
        win_prob = 1 / (1 + 10 ** (self.k * norm_diff))

        return win_prob, red_mean, blue_mean

    def attribute_match(self, red_teams: List[int], blue_teams: List[int],
                        red_actual: np.ndarray, blue_actual: np.ndarray,
                        red_pred: np.ndarray, blue_pred: np.ndarray,
                        red_weights: Optional[np.ndarray] = None,
                        blue_weights: Optional[np.ndarray] = None) -> Dict[int, np.ndarray]:
        attributions = {}
        red_err = red_actual - red_pred
        blue_err = blue_actual - blue_pred

        if red_weights is None:
            red_weights = np.full((len(red_teams), len(red_err)), 1.0 / self.num_teams)
        if blue_weights is None:
            blue_weights = np.full((len(blue_teams), len(blue_err)), 1.0 / self.num_teams)

        for j, t in enumerate(red_teams):
            current_mean = self.get_team(t).mean
            attributions[t] = current_mean + red_err * red_weights[j]
        for j, t in enumerate(blue_teams):
            current_mean = self.get_team(t).mean
            attributions[t] = current_mean + blue_err * blue_weights[j]

        return attributions

    def update_team(self, team_num: int, attribution: np.ndarray, is_elim: bool = False):
        team = self.get_team(team_num)
        n = self.counts[team_num] if team_num in self.counts else 0
        alpha = 1.0 / (1.0 + n * 0.1)
        weight = 0.33 if is_elim else 1.0
        team.add_obs(attribution, alpha, weight)
        if not is_elim:
            self.counts[team_num] = n + 1
