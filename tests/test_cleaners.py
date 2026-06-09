import unittest
import numpy as np
from backend.src.data.cleaner import CleanerRegistry
from backend.src.core.config import FTC_VECTOR_SIZE


class TestCleanerBase(unittest.TestCase):
    def setUp(self):
        self.all = {s: CleanerRegistry.get_cleaner(s) for s in ["2019", "2020", "2021", "2022", "2023", "2024", "2025"]}

    def test_all_return_32dim_vector(self):
        for sid, c in self.all.items():
            r = c.clean({"totalPointsNp": 100})
            self.assertEqual(len(r), FTC_VECTOR_SIZE, f"{sid} returned vector of len {len(r)}")

    def test_empty_data_returns_zeros(self):
        for sid, c in self.all.items():
            r = c.clean({})
            self.assertTrue(np.all(r == 0), f"{sid} non-zero on empty input")

    def test_total_maps_to_index_0(self):
        for sid, c in self.all.items():
            r = c.clean({"totalPointsNp": 42})
            self.assertEqual(r[0], 42, f"{sid} totalPointsNp not at index 0")

    def test_auto_maps_to_index_1(self):
        for sid, c in self.all.items():
            r = c.clean({"autoPoints": 35})
            self.assertEqual(r[1], 35, f"{sid} autoPoints not at index 1")

    def test_dc_maps_to_index_2(self):
        for sid, c in self.all.items():
            r = c.clean({"dcPoints": 60})
            self.assertEqual(r[2], 60, f"{sid} dcPoints not at index 2")

    def test_missing_fields_default_to_zero(self):
        c = CleanerRegistry.get_cleaner("2019")
        r = c.clean({})
        for i in range(10):
            self.assertEqual(r[i], 0, f"index {i} not zero on missing field")


class TestCleaner2019(TestCleanerBase):
    def test_endgame_maps_to_index_3(self):
        c = CleanerRegistry.get_cleaner("2019")
        r = c.clean({"egPoints": 25})
        self.assertEqual(r[3], 25)


class TestCleaner2024(TestCleanerBase):
    def test_endgame_is_composite_sum(self):
        c = CleanerRegistry.get_cleaner("2024")
        r = c.clean({"autoParkPoints": 10, "dcParkPoints": 7})
        self.assertEqual(r[3], 17)

    def test_endgame_zero_when_no_park_fields(self):
        c = CleanerRegistry.get_cleaner("2024")
        r = c.clean({"egPoints": 99})
        self.assertEqual(r[3], 0, "2024 endgame should NOT use egPoints field")

    def test_auto_teleop_not_affected_by_composite(self):
        c = CleanerRegistry.get_cleaner("2024")
        r = c.clean({"autoPoints": 20, "dcPoints": 30, "autoParkPoints": 5, "dcParkPoints": 3})
        self.assertEqual(r[1], 20)
        self.assertEqual(r[2], 30)


class TestCleaner2025(TestCleanerBase):
    def test_bool_fields_convert_to_float(self):
        c = CleanerRegistry.get_cleaner("2025")
        r = c.clean({"movementRp": True, "goalRp": True, "patternRp": True})
        self.assertEqual(r[4], 1.0, "movementRp True should be 1.0")
        self.assertEqual(r[5], 1.0, "goalRp True should be 1.0")
        self.assertEqual(r[6], 1.0, "patternRp True should be 1.0")

    def test_bool_fields_false_to_zero(self):
        c = CleanerRegistry.get_cleaner("2025")
        r = c.clean({"movementRp": False, "goalRp": False, "patternRp": False})
        self.assertEqual(r[4], 0.0)
        self.assertEqual(r[5], 0.0)
        self.assertEqual(r[6], 0.0)

    def test_field_weights_applied(self):
        c = CleanerRegistry.get_cleaner("2025")
        r = c.clean({"autoArtifactClassifiedPoints": 9, "dcArtifactClassifiedPoints": 6})
        self.assertAlmostEqual(r[7], 3.0, msg="autoArtifactClassifiedPoints should be /3")
        self.assertAlmostEqual(r[8], 2.0, msg="dcArtifactClassifiedPoints should be /3")

    def test_dcBasePoints_at_index_3(self):
        c = CleanerRegistry.get_cleaner("2025")
        r = c.clean({"dcBasePoints": 15})
        self.assertEqual(r[3], 15)

    def test_dcDepotPoints_at_index_9(self):
        c = CleanerRegistry.get_cleaner("2025")
        r = c.clean({"dcDepotPoints": 4})
        self.assertEqual(r[9], 4)

    def test_full_match_vector(self):
        c = CleanerRegistry.get_cleaner("2025")
        r = c.clean({
            "totalPointsNp": 80,
            "autoPoints": 15,
            "dcPoints": 55,
            "dcBasePoints": 10,
            "movementRp": True,
            "goalRp": True,
            "patternRp": False,
            "autoArtifactClassifiedPoints": 12,
            "dcArtifactClassifiedPoints": 30,
            "dcDepotPoints": 2,
        })
        self.assertEqual(r[0], 80)
        self.assertEqual(r[1], 15)
        self.assertEqual(r[2], 55)
        self.assertEqual(r[3], 10)
        self.assertEqual(r[4], 1.0)
        self.assertEqual(r[5], 1.0)
        self.assertEqual(r[6], 0.0)
        self.assertAlmostEqual(r[7], 4.0)
        self.assertAlmostEqual(r[8], 10.0)
        self.assertEqual(r[9], 2)


class TestAttributionWeights(unittest.TestCase):
    def test_default_equal_split(self):
        c = CleanerRegistry.get_cleaner("2019")
        w = c.get_attribution_weights({}, [101, 202])
        self.assertEqual(w.shape, (2, FTC_VECTOR_SIZE))
        self.assertTrue(np.allclose(w, 0.5))

    def test_three_teams_third_split(self):
        c = CleanerRegistry.get_cleaner("2019")
        w = c.get_attribution_weights({}, [101, 202, 303])
        self.assertEqual(w.shape, (3, FTC_VECTOR_SIZE))
        self.assertTrue(np.allclose(w, 1.0 / 3))

    def test_2025_asymmetric_weights(self):
        c = CleanerRegistry.get_cleaner("2025")
        w = c.get_attribution_weights({"dcBase1": 8, "dcBase2": 2, "autoLeave1": 1, "autoLeave2": 0,
                                       "autoPoints": 20}, [101, 202])
        self.assertEqual(w.shape, (2, FTC_VECTOR_SIZE))
        # dcBase: team 1 did 8/10, team 2 did 2/10
        self.assertAlmostEqual(w[0, 3], 0.8)
        self.assertAlmostEqual(w[1, 3], 0.2)
        # auto: 1 out of 20 is known from autoLeave, 19 unknown split equally
        # team 1: 1/20 + 19/20 * 0.5 = 0.525
        self.assertAlmostEqual(w[0, 1], 0.525)
        # team 2: 0/20 + 19/20 * 0.5 = 0.475
        self.assertAlmostEqual(w[1, 1], 0.475)

    def test_2025_no_station_data_falls_back_to_equal(self):
        c = CleanerRegistry.get_cleaner("2025")
        w = c.get_attribution_weights({"dcBase1": 0, "dcBase2": 0, "autoLeave1": 0, "autoLeave2": 0,
                                       "autoPoints": 0}, [101, 202])
        self.assertTrue(np.allclose(w, 0.5))


class TestGraphQLFragments(unittest.TestCase):
    def test_all_fragments_contain_red_and_blue(self):
        for sid in ["2019", "2020", "2021", "2022", "2023", "2024", "2025"]:
            c = CleanerRegistry.get_cleaner(sid)
            f = c.get_graphql_fragment()
            self.assertIn("red {", f, f"{sid} missing red block")
            self.assertIn("blue {", f, f"{sid} missing blue block")

    def test_single_fragment_type(self):
        for sid in ["2019", "2022", "2023", "2024", "2025"]:
            c = CleanerRegistry.get_cleaner(sid)
            f = c.get_graphql_fragment()
            self.assertIn(f"MatchScores{sid}", f, f"{sid} missing MatchScores{sid}")

    def test_dual_fragment_2020(self):
        c = CleanerRegistry.get_cleaner("2020")
        f = c.get_graphql_fragment()
        self.assertIn("MatchScores2020Trad", f)
        self.assertIn("MatchScores2020Remote", f)

    def test_dual_fragment_2021(self):
        c = CleanerRegistry.get_cleaner("2021")
        f = c.get_graphql_fragment()
        self.assertIn("MatchScores2021Trad", f)
        self.assertIn("MatchScores2021Remote", f)

    def test_api_fields_included(self):
        c = CleanerRegistry.get_cleaner("2025")
        f = c.get_graphql_fragment()
        self.assertIn("totalPointsNp", f)
        self.assertIn("movementRp", f)
        self.assertIn("autoLeave1", f)


class TestAggregate(unittest.TestCase):
    def test_aggregate_passthrough(self):
        for sid in ["2019", "2020", "2021", "2022", "2023", "2024", "2025"]:
            c = CleanerRegistry.get_cleaner(sid)
            result = c.aggregate({"a": 1, "b": 2})
            self.assertEqual(result, {"a": 1, "b": 2}, f"{sid} aggregate not passthrough")


if __name__ == "__main__":
    unittest.main()
