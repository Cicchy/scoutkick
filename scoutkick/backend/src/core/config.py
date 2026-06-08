from dataclasses import dataclass
from typing import Dict, List

FTC_VECTOR_SIZE = 32


def _make_dimensions(active: Dict[int, str]) -> List[str]:
    dims = [""] * FTC_VECTOR_SIZE
    for idx, name in active.items():
        dims[idx] = name
    return dims


@dataclass
class SeasonConfig:
    season_id: str
    active_dimensions: List[str]
    indices: List[int]
    rp_indices: List[int]
    default_mean_total: float = 20.0
    default_variance: float = 100.0
    score_sd: float = 20.0

    def __post_init__(self):
        self.dimensions = _make_dimensions(dict(zip(self.indices, self.active_dimensions)))


SEASON_CONFIGS: Dict[str, SeasonConfig] = {
    "2025": SeasonConfig(
        season_id="2025",
        active_dimensions=[
            "total", "auto", "teleop", "endgame",
            "rp1", "rp2", "rp3",
            "auto_classified", "teleop_classified", "teleop_depot",
        ],
        indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        rp_indices=[4, 5, 6],
        default_mean_total=20.0,
        default_variance=100.0,
        score_sd=20.0,
    ),
    "2024": SeasonConfig(
        season_id="2024",
        active_dimensions=["total", "auto", "teleop", "endgame"],
        indices=[0, 1, 2, 3],
        rp_indices=[],
        default_mean_total=20.0,
        default_variance=100.0,
        score_sd=20.0,
    ),
    "2023": SeasonConfig(
        season_id="2023",
        active_dimensions=["total", "auto", "teleop", "endgame"],
        indices=[0, 1, 2, 3],
        rp_indices=[],
        default_mean_total=15.0,
        default_variance=100.0,
        score_sd=20.0,
    ),
    "2022": SeasonConfig(
        season_id="2022",
        active_dimensions=["total", "auto", "teleop", "endgame"],
        indices=[0, 1, 2, 3],
        rp_indices=[],
        default_mean_total=15.0,
        default_variance=100.0,
        score_sd=20.0,
    ),
    "2021": SeasonConfig(
        season_id="2021",
        active_dimensions=["total", "auto", "teleop", "endgame"],
        indices=[0, 1, 2, 3],
        rp_indices=[],
        default_mean_total=15.0,
        default_variance=100.0,
        score_sd=20.0,
    ),
    "2020": SeasonConfig(
        season_id="2020",
        active_dimensions=["total", "auto", "teleop", "endgame"],
        indices=[0, 1, 2, 3],
        rp_indices=[],
        default_mean_total=15.0,
        default_variance=100.0,
        score_sd=20.0,
    ),
    "2019": SeasonConfig(
        season_id="2019",
        active_dimensions=["total", "auto", "teleop", "endgame"],
        indices=[0, 1, 2, 3],
        rp_indices=[],
        default_mean_total=15.0,
        default_variance=100.0,
        score_sd=20.0,
    ),
}


def get_season_config(season_id: str) -> SeasonConfig:
    if season_id not in SEASON_CONFIGS:
        raise ValueError(f"No configuration found for season {season_id}")
    return SEASON_CONFIGS[season_id]
