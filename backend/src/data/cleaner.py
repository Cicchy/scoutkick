import numpy as np
from typing import Dict, List, Any, Optional
from abc import ABC
from backend.src.core.config import FTC_VECTOR_SIZE


class BaseCleaner(ABC):
    SEASON_ID: str = ""
    GRAPHQL_FRAGMENT: str = ""

    FIELD_TO_EPA_INDEX: Dict[str, int] = {}
    BOOL_FIELDS: List[str] = []
    FIELD_WEIGHTS: Dict[str, float] = {}
    COMPOSITE_DIMS: Dict[int, List[str]] = {}
    API_FIELDS: List[str] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.API_FIELDS:
            fields = set(cls.FIELD_TO_EPA_INDEX.keys())
            for dim_fields in cls.COMPOSITE_DIMS.values():
                fields.update(dim_fields)
            cls.API_FIELDS = sorted(fields)

    def clean(self, data: Dict[str, Any]) -> np.ndarray:
        vec = np.zeros(FTC_VECTOR_SIZE)
        for field, idx in self.FIELD_TO_EPA_INDEX.items():
            val = data.get(field, 0)
            if field in self.BOOL_FIELDS:
                val = 1.0 if val else 0.0
            vec[idx] = val * self.FIELD_WEIGHTS.get(field, 1.0)
        for idx, fields in self.COMPOSITE_DIMS.items():
            vec[idx] = sum(data.get(f, 0) for f in fields)
        return vec

    def get_graphql_fragment(self) -> str:
        fields = "\n".join(f"            {f}" for f in self.API_FIELDS)
        return f"""        ... on {self.GRAPHQL_FRAGMENT} {{
          red {{
{fields}
          }}
          blue {{
{fields}
          }}
        }}"""

    def aggregate(self, scores: Dict[str, Any]) -> Dict[str, Any]:
        return scores

    def get_attribution_weights(self, raw_scores: Dict[str, Any],
                                team_numbers: List[int]) -> np.ndarray:
        n = len(team_numbers)
        return np.full((n, FTC_VECTOR_SIZE), 1.0 / n)


# ── 2025: Into The Deep ──────────────────────────────────────────

class FTC2025Cleaner(BaseCleaner):
    SEASON_ID = "2025"
    GRAPHQL_FRAGMENT = "MatchScores2025"

    API_FIELDS = [
        "totalPointsNp", "autoPoints", "dcPoints", "dcBasePoints",
        "movementRp", "goalRp", "patternRp",
        "autoArtifactClassifiedPoints", "dcArtifactClassifiedPoints",
        "dcDepotPoints", "alliance",
        "autoLeave1", "autoLeave2",
        "dcBase1", "dcBase2",
    ]

    BOOL_FIELDS = ["movementRp", "goalRp", "patternRp"]

    FIELD_WEIGHTS = {
        "autoArtifactClassifiedPoints": 1.0 / 3.0,
        "dcArtifactClassifiedPoints": 1.0 / 3.0,
    }

    FIELD_TO_EPA_INDEX = {
        "totalPointsNp": 0,
        "autoPoints": 1,
        "dcPoints": 2,
        "dcBasePoints": 3,
        "movementRp": 4,
        "goalRp": 5,
        "patternRp": 6,
        "autoArtifactClassifiedPoints": 7,
        "dcArtifactClassifiedPoints": 8,
        "dcDepotPoints": 9,
    }

    def get_attribution_weights(self, raw_scores: Dict[str, Any],
                                team_numbers: List[int]) -> np.ndarray:
        n = len(team_numbers)
        weights = np.full((n, FTC_VECTOR_SIZE), 1.0 / n)
        dc1 = float(raw_scores.get("dcBase1", 0))
        dc2 = float(raw_scores.get("dcBase2", 0))
        dc_total = dc1 + dc2
        if dc_total > 0:
            weights[0, 3] = dc1 / dc_total
            weights[1, 3] = dc2 / dc_total
        al1 = float(raw_scores.get("autoLeave1", 0))
        al2 = float(raw_scores.get("autoLeave2", 0))
        auto_total = float(raw_scores.get("autoPoints", 0))
        auto_leave_total = al1 + al2
        if auto_total > 0 and auto_leave_total > 0:
            known_share = auto_leave_total / auto_total
            unknown_share = 1.0 - known_share
            weights[0, 1] = (al1 / auto_total) + unknown_share * 0.5
            weights[1, 1] = (al2 / auto_total) + unknown_share * 0.5
        return weights


# ── 2024: Into The Deep (same name — prev season) ───────────────

class FTC2024Cleaner(BaseCleaner):
    SEASON_ID = "2024"
    GRAPHQL_FRAGMENT = "MatchScores2024"

    API_FIELDS = [
        "totalPointsNp", "autoPoints", "dcPoints",
        "autoParkPoints", "dcParkPoints",
        "autoSamplePoints", "autoSpecimenPoints",
        "dcSamplePoints", "dcSpecimenPoints",
    ]

    FIELD_TO_EPA_INDEX = {
        "totalPointsNp": 0,
        "autoPoints": 1,
        "dcPoints": 2,
    }

    COMPOSITE_DIMS = {3: ["autoParkPoints", "dcParkPoints"]}


# ── 2023: CENTERSTAGE ────────────────────────────────────────────

class FTC2023Cleaner(BaseCleaner):
    SEASON_ID = "2023"
    GRAPHQL_FRAGMENT = "MatchScores2023"

    API_FIELDS = [
        "totalPointsNp", "autoPoints", "dcPoints", "egPoints",
        "autoNavPoints", "autoPixelPoints", "purplePoints", "yellowPoints",
        "egNavPoints", "dronePoints", "setLinePoints", "mosaicPoints",
    ]

    FIELD_TO_EPA_INDEX = {
        "totalPointsNp": 0,
        "autoPoints": 1,
        "dcPoints": 2,
        "egPoints": 3,
    }


# ── 2022: Power Play ────────────────────────────────────────────

class FTC2022Cleaner(BaseCleaner):
    SEASON_ID = "2022"
    GRAPHQL_FRAGMENT = "MatchScores2022"

    API_FIELDS = [
        "totalPointsNp", "autoPoints", "dcPoints", "egPoints",
        "autoNavPoints", "autoConePoints",
        "egNavPoints", "ownershipPoints", "circuitPoints",
    ]

    FIELD_TO_EPA_INDEX = {
        "totalPointsNp": 0,
        "autoPoints": 1,
        "dcPoints": 2,
        "egPoints": 3,
    }


# ── 2021: Freight Frenzy ─────────────────────────────────────────

class FTC2021Cleaner(BaseCleaner):
    SEASON_ID = "2021"
    GRAPHQL_FRAGMENT = ""

    TRAD_FIELDS = [
        "totalPointsNp", "autoPoints", "dcPoints", "egPoints",
        "autoCarouselPoints", "autoNavPoints", "autoFreightPoints", "autoBonusPoints",
        "dcAllianceHubPoints", "dcSharedHubPoints", "dcStoragePoints",
        "egDuckPoints", "allianceBalancedPoints", "sharedUnbalancedPoints",
        "egParkPoints", "cappingPoints",
    ]

    REMOTE_FIELDS = [
        "totalPointsNp", "autoPoints", "dcPoints", "egPoints",
        "autoCarouselPoints", "autoNavPoints", "autoFreightPoints", "autoBonusPoints",
        "dcAllianceHubPoints", "dcStoragePoints",
        "egDuckPoints", "allianceBalancedPoints",
        "egParkPoints", "cappingPoints",
    ]

    FIELD_TO_EPA_INDEX = {
        "totalPointsNp": 0,
        "autoPoints": 1,
        "dcPoints": 2,
        "egPoints": 3,
    }

    def get_graphql_fragment(self) -> str:
        trad = "\n".join(f"            {f}" for f in self.TRAD_FIELDS)
        remote = "\n".join(f"            {f}" for f in self.REMOTE_FIELDS)
        return f"""        ... on MatchScores2021Trad {{
          red {{
{trad}
          }}
          blue {{
{trad}
          }}
        }}
        ... on MatchScores2021Remote {{
          alliance
{remote}
        }}"""


# ── 2020: Ultimate Goal ──────────────────────────────────────────

class FTC2020Cleaner(BaseCleaner):
    SEASON_ID = "2020"
    GRAPHQL_FRAGMENT = ""

    TRAD_FIELDS = [
        "totalPointsNp", "autoPoints", "dcPoints", "egPoints",
        "autoNavPoints", "autoTowerPoints", "autoWobblePoints", "autoPowershotPoints",
        "egWobblePoints", "egPowershotPoints", "egWobbleRingPoints",
    ]

    REMOTE_FIELDS = [
        "totalPointsNp", "autoPoints", "dcPoints", "egPoints",
        "autoNavPoints", "autoTowerPoints", "autoWobblePoints", "autoPowershotPoints",
        "egWobblePoints", "egPowershotPoints", "egWobbleRingPoints",
    ]

    FIELD_TO_EPA_INDEX = {
        "totalPointsNp": 0,
        "autoPoints": 1,
        "dcPoints": 2,
        "egPoints": 3,
    }

    def get_graphql_fragment(self) -> str:
        trad = "\n".join(f"            {f}" for f in self.TRAD_FIELDS)
        remote = "\n".join(f"            {f}" for f in self.REMOTE_FIELDS)
        return f"""        ... on MatchScores2020Trad {{
          red {{
{trad}
          }}
          blue {{
{trad}
          }}
        }}
        ... on MatchScores2020Remote {{
          alliance
{remote}
        }}"""


# ── 2019: SkyStone ───────────────────────────────────────────────

class FTC2019Cleaner(BaseCleaner):
    SEASON_ID = "2019"
    GRAPHQL_FRAGMENT = "MatchScores2019"

    API_FIELDS = [
        "totalPointsNp", "autoPoints", "dcPoints", "egPoints",
        "autoNavPoints", "autoRepositioningPoints", "autoDeliveryPoints", "autoPlacementPoints",
        "dcDeliveryPoints", "dcPlacementPoints",
        "skyscraperBonusPoints", "cappingPoints",
        "egParkPoints", "egFoundationMovedPoints",
    ]

    FIELD_TO_EPA_INDEX = {
        "totalPointsNp": 0,
        "autoPoints": 1,
        "dcPoints": 2,
        "egPoints": 3,
    }


# ── Registry ─────────────────────────────────────────────────────

class CleanerRegistry:
    _registry: Dict[str, BaseCleaner] = {}

    @classmethod
    def register(cls, season_id: str, cleaner: BaseCleaner):
        cls._registry[season_id] = cleaner

    @classmethod
    def get_cleaner(cls, season_id: str) -> BaseCleaner:
        if season_id not in cls._registry:
            raise ValueError(f"No cleaner registered for season {season_id}")
        return cls._registry[season_id]


CleanerRegistry.register("2025", FTC2025Cleaner())
CleanerRegistry.register("2024", FTC2024Cleaner())
CleanerRegistry.register("2023", FTC2023Cleaner())
CleanerRegistry.register("2022", FTC2022Cleaner())
CleanerRegistry.register("2021", FTC2021Cleaner())
CleanerRegistry.register("2020", FTC2020Cleaner())
CleanerRegistry.register("2019", FTC2019Cleaner())
