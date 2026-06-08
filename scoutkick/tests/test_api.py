import unittest
import urllib.request
import urllib.error
import json
import subprocess
import sys
import time
import os

API_BASE = "http://127.0.0.1:8000"


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scoutkick/
_PARENT_ROOT = os.path.dirname(_REPO_ROOT)  # EPA FTC/


class WithServer(unittest.TestCase):
    server_process = None

    @classmethod
    def setUpClass(cls):
        env = {**os.environ, "PYTHONPATH": _PARENT_ROOT}
        cls.server_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.main:app",
             "--host", "127.0.0.1", "--port", "8000"],
            cwd=_REPO_ROOT, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        for _ in range(50):
            try:
                urllib.request.urlopen(f"{API_BASE}/", timeout=1)
                return
            except Exception:
                time.sleep(0.2)
        raise RuntimeError("Server did not start in time")

    @classmethod
    def tearDownClass(cls):
        if cls.server_process:
            cls.server_process.terminate()
            cls.server_process.wait(timeout=5)


class TestAPIHealth(WithServer):
    def test_root_returns_name(self):
        resp = json.loads(urllib.request.urlopen(f"{API_BASE}/").read())
        self.assertEqual(resp["name"], "scoutkick")

    def test_docs_available(self):
        resp = urllib.request.urlopen(f"{API_BASE}/docs")
        self.assertEqual(resp.status, 200)


class TestAPITeams(WithServer):
    def test_list_teams_returns_value_and_count(self):
        resp = json.loads(urllib.request.urlopen(f"{API_BASE}/v1/teams?season=2025&limit=5").read())
        self.assertIn("value", resp)
        self.assertIn("count", resp)
        self.assertLessEqual(len(resp["value"]), 5)

    def test_list_teams_sorts_by_metric(self):
        resp = json.loads(urllib.request.urlopen(
            f"{API_BASE}/v1/teams?season=2025&metric=total&ascending=true&limit=5").read())
        values = [t["total"] for t in resp["value"] if t["total"] is not None]
        self.assertEqual(sorted(values), values)

    def test_get_team_returns_team_matches(self):
        teams = json.loads(urllib.request.urlopen(f"{API_BASE}/v1/teams?season=2025&limit=1").read())
        if teams["value"]:
            t = teams["value"][0]["team"]
            resp = json.loads(urllib.request.urlopen(f"{API_BASE}/v1/team/{t}?season=2025").read())
            self.assertIn("team_matches", resp)
            self.assertEqual(resp["total"], resp["mean"][0])

    def test_get_team_404(self):
        try:
            urllib.request.urlopen(f"{API_BASE}/v1/team/99999999?season=2025")
            self.fail("Expected 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)

    def test_team_matches_returns_matches(self):
        teams = json.loads(urllib.request.urlopen(f"{API_BASE}/v1/teams?season=2025&limit=1").read())
        if teams["value"]:
            t = teams["value"][0]["team"]
            resp = json.loads(urllib.request.urlopen(f"{API_BASE}/v1/team/{t}/matches?season=2025&limit=2").read())
            self.assertIn("value", resp)
            if resp["value"]:
                m = resp["value"][0]
                self.assertIn("epa_pre", m)
                self.assertIn("epa_post", m)
                self.assertIn("win_prob", m)


class TestAPIEvents(WithServer):
    def test_list_events(self):
        resp = json.loads(urllib.request.urlopen(f"{API_BASE}/v1/events?season=2025&limit=3").read())
        self.assertIn("value", resp)
        self.assertIn("count", resp)

    def test_get_event_returns_teams(self):
        resp = json.loads(urllib.request.urlopen(f"{API_BASE}/v1/events?season=2025&limit=1").read())
        if resp["value"]:
            ec = resp["value"][0]["event_code"]
            event = json.loads(urllib.request.urlopen(f"{API_BASE}/v1/event/{ec}?season=2025").read())
            self.assertIn("teams", event)
            self.assertIn("team_count", event)

    def test_get_event_404(self):
        try:
            urllib.request.urlopen(f"{API_BASE}/v1/event/NONEXISTENT?season=2025")
            self.fail("Expected 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)

    def test_event_matches(self):
        resp = json.loads(urllib.request.urlopen(f"{API_BASE}/v1/events?season=2025&limit=1").read())
        if resp["value"]:
            ec = resp["value"][0]["event_code"]
            resp2 = json.loads(urllib.request.urlopen(f"{API_BASE}/v1/event/{ec}/matches?season=2025").read())
            self.assertIn("value", resp2)


class TestAPISeasons(WithServer):
    def test_list_seasons(self):
        resp = json.loads(urllib.request.urlopen(f"{API_BASE}/v1/seasons").read())
        self.assertIsInstance(resp, list)

    def test_get_season(self):
        resp = json.loads(urllib.request.urlopen(f"{API_BASE}/v1/season/2025").read())
        self.assertIn("score_sd", resp)
        self.assertIn("num_teams", resp)
        self.assertIn("num_matches", resp)

    def test_get_season_404(self):
        try:
            urllib.request.urlopen(f"{API_BASE}/v1/season/2099")
            self.fail("Expected 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)


class TestAPIPredict(WithServer):
    def test_predict_returns_win_prob(self):
        resp = json.loads(urllib.request.urlopen(
            f"{API_BASE}/v1/predict?red=3491,32736&blue=23400,24599&season=2025").read())
        self.assertIn("red_win_prob", resp)
        self.assertIn("blue_win_prob", resp)
        self.assertAlmostEqual(resp["red_win_prob"] + resp["blue_win_prob"], 1.0)

    def test_predict_returns_team_details(self):
        resp = json.loads(urllib.request.urlopen(
            f"{API_BASE}/v1/predict?red=3491,32736&blue=23400,24599&season=2025").read())
        self.assertEqual(len(resp["red_teams"]), 2)
        self.assertEqual(len(resp["blue_teams"]), 2)
        self.assertIn("total", resp["red_teams"][0])
        self.assertIn("norm_epa", resp["red_teams"][0])

    def test_predict_400_for_wrong_team_count(self):
        try:
            urllib.request.urlopen(f"{API_BASE}/v1/predict?red=3491&blue=23400,24599&season=2025")
            self.fail("Expected 400")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)

    def test_compare_teams(self):
        resp = json.loads(urllib.request.urlopen(
            f"{API_BASE}/v1/compare?teams=3491,32736,23400&season=2025").read())
        self.assertIn("teams", resp)
        self.assertEqual(len(resp["teams"]), 3)


class TestAPIMatches(WithServer):
    def test_list_matches_returns_matches(self):
        resp = json.loads(urllib.request.urlopen(f"{API_BASE}/v1/matches?season=2025&limit=3").read())
        self.assertIn("value", resp)
        if resp["value"]:
            m = resp["value"][0]
            self.assertIn("event_code", m)
            self.assertIn("match_id", m)

    def test_get_match_returns_teams(self):
        resp = json.loads(urllib.request.urlopen(f"{API_BASE}/v1/matches?season=2025&limit=1").read())
        if resp["value"]:
            ec = resp["value"][0]["event_code"]
            mid = resp["value"][0]["match_id"]
            match = json.loads(urllib.request.urlopen(
                f"{API_BASE}/v1/match/{ec}/{mid}?season=2025").read())
            self.assertIn("teams", match)
            self.assertGreater(len(match["teams"]), 0)


if __name__ == "__main__":
    unittest.main()
