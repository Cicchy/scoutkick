import unittest
import tempfile
import os
import numpy as np
from backend.src.storage.sqlite_storage import SQLiteStorage


class TestSQLiteStorage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.season = "2025"
        self.storage = SQLiteStorage(self.tmp.name, self.season)

    def tearDown(self):
        os.unlink(self.tmp.name)

    # ── team_seasons ──

    def test_save_and_load_team(self):
        mean = np.array([80.0, 25.0, 40.0, 15.0] + [0.0] * 28)
        var = np.array([100.0] * 32)
        self.storage.save_team(101, mean, var, 0.5, 10.0, 8, norm_epa=1500.0)
        loaded = self.storage.load_team(101)
        self.assertIsNotNone(loaded)
        np.testing.assert_array_almost_equal(loaded["mean"], mean)
        np.testing.assert_array_almost_equal(loaded["var"], var)
        self.assertEqual(loaded["skew"], 0.5)
        self.assertEqual(loaded["n"], 10.0)
        self.assertEqual(loaded["count"], 8)
        self.assertEqual(loaded["norm_epa"], 1500.0)

    def test_load_missing_team_returns_none(self):
        loaded = self.storage.load_team(99999)
        self.assertIsNone(loaded)

    def test_save_all_teams_and_load_all(self):
        sn_mean = np.array([80.0, 25.0, 40.0, 15.0] + [0.0] * 28)
        sn_var = np.array([100.0] * 32)

        class FakeSN:
            mean = sn_mean
            var = sn_var
            skew = 0.3
            n = 5.0

        epas = {101: FakeSN(), 102: FakeSN()}
        counts = {101: 5, 102: 7}
        norm_epas = {101: 1500.0, 102: 1400.0}
        self.storage.save_all_teams(epas, counts, norm_epas)

        all_teams = self.storage.load_all_teams()
        self.assertEqual(len(all_teams), 2)
        self.assertAlmostEqual(all_teams[101]["mean"][0], 80.0)
        self.assertEqual(all_teams[102]["norm_epa"], 1400.0)

    def test_load_team_cross_season(self):
        mean = np.array([80.0] + [0.0] * 31)
        var = np.array([100.0] * 32)
        self.storage.save_team(101, mean, var, 0.0, 1.0, 5)
        storage2 = SQLiteStorage(self.tmp.name, "2024")
        storage2.save_team(101, mean, var, 0.0, 1.0, 3)
        cross = self.storage.load_team_cross_season(101)
        self.assertEqual(len(cross), 2)

    def test_delete_season(self):
        mean = np.array([80.0] + [0.0] * 31)
        var = np.array([100.0] * 32)
        self.storage.save_team(101, mean, var, 0.0, 1.0, 5)
        self.storage.delete_season()
        self.assertIsNone(self.storage.load_team(101))

    # ── team_events ──

    def test_save_and_load_team_event(self):
        self.storage.save_team_event(101, "EVENT1", epa_start=80.0, epa_max=90.0,
                                      epa_mean=85.0, count=10, norm_epa=1500.0,
                                      event_type="Qualifier")
        events = self.storage.load_team_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["team"], 101)
        self.assertEqual(events[0]["event_code"], "EVENT1")
        self.assertEqual(events[0]["epa_start"], 80.0)
        self.assertEqual(events[0]["epa_max"], 90.0)

    def test_load_team_event_specific(self):
        self.storage.save_team_event(101, "EVENT1")
        self.storage.save_team_event(101, "EVENT2")
        event = self.storage.load_team_event(101, "EVENT1")
        self.assertIsNotNone(event)
        self.assertEqual(event["event_code"], "EVENT1")

    # ── team_matches ──

    def test_save_and_load_team_match(self):
        self.storage.save_team_match(101, "EVENT1", "1", epa_pre=80.0, epa_post=82.5,
                                      win_prob=0.65, is_elim=False)
        matches = self.storage.load_team_matches(101)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["epa_pre"], 80.0)
        self.assertEqual(matches[0]["epa_post"], 82.5)
        self.assertEqual(matches[0]["win_prob"], 0.65)
        self.assertEqual(matches[0]["is_elim"], 0)

    def test_multiple_matches_ordered_by_processed_at(self):
        self.storage.save_team_match(101, "EVENT1", "1")
        self.storage.save_team_match(101, "EVENT1", "2")
        self.storage.save_team_match(101, "EVENT2", "1")
        matches = self.storage.load_team_matches(101)
        self.assertEqual(len(matches), 3)

    def test_get_processed_match_keys(self):
        self.storage.save_team_match(101, "EVENT1", "1")
        self.storage.save_team_match(102, "EVENT1", "1")
        self.storage.save_team_match(101, "EVENT1", "2")
        keys = self.storage.get_processed_match_keys()
        self.assertEqual(len(keys), 2)
        self.assertIn(("EVENT1", "1"), keys)
        self.assertIn(("EVENT1", "2"), keys)

    def test_upsert_updates_post_epa(self):
        self.storage.save_team_match(101, "EVENT1", "1", epa_pre=80.0, epa_post=82.5)
        self.storage.save_team_match(101, "EVENT1", "1", epa_pre=80.0, epa_post=90.0)
        matches = self.storage.load_team_matches(101)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["epa_post"], 90.0)
        self.assertEqual(matches[0]["epa_pre"], 80.0)

    # ── seasons ──

    def test_save_and_load_season_meta(self):
        comp_means = np.array([50.0, 20.0, 30.0, 10.0] + [0.0] * 28)
        self.storage.save_season_meta(score_mean=100.0, score_sd=50.0,
                                       component_means=comp_means,
                                       num_matches=1000, num_teams=500)
        meta = self.storage.load_season_meta()
        self.assertIsNotNone(meta)
        self.assertEqual(meta["score_mean"], 100.0)
        self.assertEqual(meta["score_sd"], 50.0)
        self.assertEqual(meta["num_matches"], 1000)
        self.assertEqual(meta["num_teams"], 500)
        np.testing.assert_array_almost_equal(meta["component_means"], comp_means)

    def test_load_missing_season_meta_returns_none(self):
        meta = self.storage.load_season_meta()
        self.assertIsNone(meta)


class TestPriorSeasons(unittest.TestCase):
    def test_get_prior_seasons_returns_4_years(self):
        priors = SQLiteStorage.get_prior_seasons("2025")
        self.assertEqual(priors, ["2024", "2023", "2022", "2021"])

    def test_get_prior_seasons_returns_fewer_for_early_year(self):
        priors = SQLiteStorage.get_prior_seasons("2019")
        self.assertEqual(len(priors), 4)

    def test_get_prior_seasons_non_numeric(self):
        priors = SQLiteStorage.get_prior_seasons("abc")
        self.assertEqual(priors, [])

    def test_get_prior_seasons_look_back(self):
        priors = SQLiteStorage.get_prior_seasons("2025", look_back=2)
        self.assertEqual(priors, ["2024", "2023"])


class TestTeamEventsWithArrays(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.season = "2025"
        self.storage = SQLiteStorage(self.tmp.name, self.season)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_save_event_with_mean_var_arrays(self):
        mean = np.array([80.0] + [0.0] * 31)
        var = np.array([100.0] * 32)
        self.storage.save_team_event(101, "EVENT1", epa_start=80.0, epa_mean=85.0,
                                      mean=mean, var=var, skew=0.5, n=10.0, count=5)
        event = self.storage.load_team_event(101, "EVENT1")
        np.testing.assert_array_almost_equal(event["mean"], mean)
        np.testing.assert_array_almost_equal(event["var"], var)
        self.assertEqual(event["skew"], 0.5)

    def test_save_event_without_arrays(self):
        self.storage.save_team_event(101, "EVENT1", epa_start=80.0)
        event = self.storage.load_team_event(101, "EVENT1")
        self.assertIsNone(event["mean"])
        self.assertIsNone(event["var"])


if __name__ == "__main__":
    unittest.main()
