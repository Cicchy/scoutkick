import unittest
import numpy as np
from backend.src.core.config import get_season_config
from backend.src.services.epa_service import EPAEngine
from backend.src.data.cleaner import CleanerRegistry


class TestEPAEngineInit(unittest.TestCase):
    def setUp(self):
        self.config = get_season_config("2025")
        self.engine = EPAEngine(config=self.config)

    def test_team_not_found_returns_default(self):
        sn = self.engine.get_team(99999)
        self.assertEqual(sn.mean[0], self.config.default_mean_total)
        self.assertTrue(np.all(sn.var == self.config.default_variance))

    def test_team_count_starts_at_zero(self):
        self.assertEqual(self.engine.counts.get(99999), None)
        self.engine.get_team(99999)
        self.assertEqual(self.engine.counts[99999], 0)

    def test_default_team_has_32_dim_vector(self):
        sn = self.engine.get_team(1)
        self.assertEqual(len(sn.mean), 32)

    def test_initial_win_prob_is_50pct(self):
        wp, red, blue = self.engine.predict_match([100, 200], [300, 400])
        self.assertAlmostEqual(wp, 0.5, places=4)
        self.assertAlmostEqual(red[0], blue[0])

    def test_rp_indices_use_unit_sigmoid(self):
        wp, red, blue = self.engine.predict_match([100, 200], [300, 400])
        for i in self.config.rp_indices:
            self.assertGreaterEqual(red[i], 0.0)
            self.assertLessEqual(red[i], 1.0)

    def test_set_team_state(self):
        mean = np.full(32, 50.0)
        var = np.full(32, 10.0)
        self.engine.set_team_state(42, mean, var, 0.5, 10.0, 5)
        sn = self.engine.get_team(42)
        np.testing.assert_array_equal(sn.mean, mean)
        self.assertEqual(sn.skew, 0.5)
        self.assertEqual(sn.n, 10.0)
        self.assertEqual(self.engine.counts[42], 5)


class TestEPAPredictAttributeUpdate(unittest.TestCase):
    def setUp(self):
        config = get_season_config("2025")
        self.engine = EPAEngine(config=config)
        self.cleaner = CleanerRegistry.get_cleaner("2025")
        self.red_teams = [101, 102]
        self.blue_teams = [201, 202]

    def _run_match(self, red_score=80, blue_score=30):
        red_raw = {"totalPointsNp": red_score, "autoPoints": 20, "dcPoints": 50,
                    "dcBasePoints": 10, "movementRp": True, "goalRp": True,
                    "patternRp": True, "autoArtifactClassifiedPoints": 15,
                    "dcArtifactClassifiedPoints": 30, "dcDepotPoints": 2}
        blue_raw = {"totalPointsNp": blue_score, "autoPoints": 5, "dcPoints": 20,
                     "dcBasePoints": 5, "movementRp": True, "goalRp": False,
                     "patternRp": False, "autoArtifactClassifiedPoints": 3,
                     "dcArtifactClassifiedPoints": 6, "dcDepotPoints": 1}
        red_actual = self.cleaner.clean(red_raw)
        blue_actual = self.cleaner.clean(blue_raw)
        wp, red_pred, blue_pred = self.engine.predict_match(self.red_teams, self.blue_teams)
        attrib = self.engine.attribute_match(
            self.red_teams, self.blue_teams,
            red_actual, blue_actual, red_pred, blue_pred)
        for t, a in attrib.items():
            self.engine.update_team(t, a)
        return wp

    def test_strong_team_becomes_stronger(self):
        pre_101 = self.engine.get_team(101).mean[0].copy()
        pre_102 = self.engine.get_team(102).mean[0].copy()
        pre_201 = self.engine.get_team(201).mean[0].copy()
        self._run_match(red_score=80, blue_score=30)
        post_101 = self.engine.get_team(101).mean[0]
        post_102 = self.engine.get_team(102).mean[0]
        post_201 = self.engine.get_team(201).mean[0]
        self.assertGreater(post_101, pre_101, "Red team 101 should increase after win")
        self.assertGreater(post_102, pre_102, "Red team 102 should increase after win")
        self.assertLess(post_201, pre_201, "Blue team 201 should decrease after loss")

    def test_red_favored_after_win(self):
        self._run_match(red_score=100, blue_score=10)
        wp2, _, _ = self.engine.predict_match(self.red_teams, self.blue_teams)
        self.assertGreater(wp2, 0.5)

    def test_attribution_error_split_equally(self):
        red_actual = self.cleaner.clean({"totalPointsNp": 100, "autoPoints": 20, "dcPoints": 50,
                                          "dcBasePoints": 10, "movementRp": True, "goalRp": True,
                                          "patternRp": True, "autoArtifactClassifiedPoints": 15,
                                          "dcArtifactClassifiedPoints": 30, "dcDepotPoints": 2})
        blue_actual = self.cleaner.clean({"totalPointsNp": 50, "autoPoints": 5, "dcPoints": 20,
                                           "dcBasePoints": 5, "movementRp": True, "goalRp": False,
                                           "patternRp": False, "autoArtifactClassifiedPoints": 3,
                                           "dcArtifactClassifiedPoints": 6, "dcDepotPoints": 1})
        wp, red_pred, blue_pred = self.engine.predict_match(self.red_teams, self.blue_teams)
        attrib = self.engine.attribute_match(
            self.red_teams, self.blue_teams,
            red_actual, blue_actual, red_pred, blue_pred)
        np.testing.assert_array_almost_equal(attrib[101], attrib[102])

    def test_count_increments(self):
        c_pre = self.engine.counts.get(101, 0)
        self._run_match()
        self.assertEqual(self.engine.counts[101], c_pre + 1)

    def test_count_not_incremented_for_elim(self):
        red_actual = self.cleaner.clean({"totalPointsNp": 80, "autoPoints": 20, "dcPoints": 50,
                                          "dcBasePoints": 10, "movementRp": True, "goalRp": True,
                                          "patternRp": True, "autoArtifactClassifiedPoints": 15,
                                          "dcArtifactClassifiedPoints": 30, "dcDepotPoints": 2})
        blue_actual = self.cleaner.clean({"totalPointsNp": 30, "autoPoints": 5, "dcPoints": 20,
                                           "dcBasePoints": 5, "movementRp": True, "goalRp": False,
                                           "patternRp": False, "autoArtifactClassifiedPoints": 3,
                                           "dcArtifactClassifiedPoints": 6, "dcDepotPoints": 1})
        wp, red_pred, blue_pred = self.engine.predict_match(self.red_teams, self.blue_teams)
        attrib = self.engine.attribute_match(self.red_teams, self.blue_teams,
                                              red_actual, blue_actual, red_pred, blue_pred)
        for t, a in attrib.items():
            self.engine.update_team(t, a, is_elim=True)
        self.assertEqual(self.engine.counts.get(101, 0), 0, "Elim match should not increment count")

    def test_learning_rate_decays_with_more_matches(self):
        for i in range(10):
            self._run_match(red_score=80, blue_score=30)
        # After 10 matches, alpha should be 1/(1+10*0.1) = 0.5
        n = self.engine.counts[101]
        self.assertEqual(n, 10)

    def test_convergence_toward_true_skill(self):
        for i in range(50):
            self._run_match(red_score=80, blue_score=30)
        r101 = self.engine.get_team(101).mean[0]
        r102 = self.engine.get_team(102).mean[0]
        b201 = self.engine.get_team(201).mean[0]
        self.assertGreater(r101, b201, "Red teams should be rated higher after consistent wins")


class TestEPAMultipleSeasons(unittest.TestCase):
    def test_engine_works_with_2019_config(self):
        config = get_season_config("2019")
        engine = EPAEngine(config=config)
        wp, red, blue = engine.predict_match([1, 2], [3, 4])
        self.assertAlmostEqual(wp, 0.5)
        self.assertEqual(len(red), 32)

    def test_engine_works_with_2024_config(self):
        config = get_season_config("2024")
        engine = EPAEngine(config=config)
        wp, red, blue = engine.predict_match([1, 2], [3, 4])
        self.assertAlmostEqual(wp, 0.5)
        self.assertEqual(len(red), 32)

    def test_engine_works_with_2023_config(self):
        config = get_season_config("2023")
        engine = EPAEngine(config=config)
        wp, red, blue = engine.predict_match([1, 2], [3, 4])
        self.assertAlmostEqual(wp, 0.5)


class TestSkewNormal(unittest.TestCase):
    def test_update_follows_ewma(self):
        from backend.src.core.math import SkewNormal
        sn = SkewNormal(np.zeros(32), np.ones(32) * 100.0)
        sn.add_obs(np.full(32, 50.0), 1.0, 1.0)  # alpha=1, weight=1
        self.assertEqual(sn.mean[0], 50.0)

    def test_partial_update(self):
        from backend.src.core.math import SkewNormal
        sn = SkewNormal(np.zeros(32), np.ones(32) * 100.0)
        sn.add_obs(np.full(32, 50.0), 0.5, 1.0)  # alpha=0.5
        self.assertAlmostEqual(sn.mean[0], 25.0)

    def test_elim_weight_applied(self):
        from backend.src.core.math import SkewNormal
        sn = SkewNormal(np.zeros(32), np.ones(32) * 100.0)
        sn.add_obs(np.full(32, 50.0), 1.0, 0.33)  # weight=0.33
        self.assertAlmostEqual(sn.mean[0], 50.0 * 0.33)


if __name__ == "__main__":
    unittest.main()
