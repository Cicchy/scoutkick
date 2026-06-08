import numpy as np

MAX_SKEW = 0.95

def unit_sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-4 * (x - 0.5)))

def inv_unit_sigmoid(x: float) -> float:
    return 0.5 + np.log(x / (1 - x)) / 4

class SkewNormal:
    def __init__(self, mean: np.ndarray, var: np.ndarray, skew_index: int = 0):
        self.mean = mean
        self.var = var
        self.skew = 0.0
        self.n = 1.0
        self.skew_i = skew_index

    @staticmethod
    def update_mean(mean: np.ndarray, x: np.ndarray, alpha: float) -> np.ndarray:
        return (1 - alpha) * mean + alpha * x

    @staticmethod
    def update_var(var: np.ndarray, mean: np.ndarray, new_mean: np.ndarray, x: np.ndarray, alpha: float) -> np.ndarray:
        new_var = (x - mean) * (x - new_mean)
        return (1 - alpha) * var + alpha * new_var

    @staticmethod
    def update_skew(skew: float, new_var: float, mean: float, new_mean: float, x: float, alpha: float) -> float:
        if new_var == 0: return skew
        new_skew = (x - mean) * (x - new_mean)**2 / (new_var ** 1.5)
        new_skew = (1 - alpha) * skew + alpha * new_skew
        return min(max(new_skew, -MAX_SKEW), MAX_SKEW)

    def add_obs(self, x: np.ndarray, percent: float, weight: float) -> None:
        new_mean = self.update_mean(self.mean, x, percent)
        new_var = self.update_var(self.var, self.mean, new_mean, x, percent)
        new_skew = self.update_skew(
            self.skew,
            new_var[self.skew_i],
            self.mean[self.skew_i],
            new_mean[self.skew_i],
            x[self.skew_i],
            percent
        )
        new_n = self.n * (1 - percent) + 1
        self.mean = weight * new_mean + (1 - weight) * self.mean
        self.var = new_var * weight + (1 - weight) * self.var
        self.skew = new_skew * weight + (1 - weight) * self.skew
        self.n = new_n * weight + (1 - weight) * self.n

    def __repr__(self) -> str:
        return f"SkewNormal(mean={self.mean}, var={self.var}, skew={self.skew}, n={self.n})"
