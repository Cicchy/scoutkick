import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from backend.src.services.pipeline_service import EPAPipeline
from backend.src.storage.sqlite_storage import SQLiteStorage

DB_PATH = "cache/epa_data.db"


def safe_load():
    storage = SQLiteStorage(DB_PATH, "2025")
    meta = storage.load_season_meta()
    if meta and storage.load_all_teams():
        print(f"Loaded from DB ({meta['num_teams']} teams, {meta['num_matches']} matches)")
        return storage, meta
    return None, None


def run_pipeline():
    pipeline = EPAPipeline("2025", db_path=DB_PATH, calibrate=True)
    engine = pipeline.run()
    storage = pipeline.storage
    meta = storage.load_season_meta()
    return storage, meta, engine


def predict_win_prob(engine, team_a, team_b):
    z = (team_a - team_b) / engine.score_sd
    k = engine.k
    return 1 / (1 + 10 ** (k * z))


def show_team(storage, engine, meta, team_num):
    sn = engine.get_team(team_num)
    norm = storage.load_season_teams("2025").get(team_num, {}).get("norm_epa", None)
    print(f"\n{'='*50}")
    print(f"  Team {team_num}")
    print(f"{'='*50}")
    print(f"  norm_epa:      {norm:>8.1f}" if norm else "")
    print(f"  Total EPA:     {sn.mean[0]:>8.1f}  (points per match, alliance avg={meta['score_mean']:.0f})")
    print(f"  Auto EPA:      {sn.mean[1]:>8.1f}")
    print(f"  Teleop EPA:    {sn.mean[2]:>8.1f}")
    print(f"  Endgame EPA:   {sn.mean[3]:>8.1f}")
    if meta:
        dims = ["Total", "Auto", "Teleop", "Endgame"]
        idx = ["RP (Win)", "RP (Score)", "RP (Penalty)", "Auto Class.", "Teleop Class.", "Teleop Depot"]
        dims += idx
        print(f"\n  Component breakdown:")
        for i, name in enumerate(dims):
            if i < len(sn.mean):
                pct = sn.mean[i] / meta["score_mean"] * 100 if meta["score_mean"] > 0 else 0
                print(f"    {name:17s}  {sn.mean[i]:>8.1f}  ({pct:5.1f}% of alliance avg)")
    print(f"  Variance:      {sn.var[0]:>8.1f}")
    print(f"  Skewness:      {sn.skew:>8.1f}")
    print(f"  Match count:   {engine.counts.get(team_num, 0):>8d}")

    team_matches = storage.load_team_matches(team_num)
    if team_matches:
        print(f"\n  Match history ({len(team_matches)} matches):")
        for m in team_matches[-5:]:
            elim = "E" if m["is_elim"] else "Q"
            wp = m["win_prob"]
            epa_pre = m["epa_pre"]
            epa_post = m["epa_post"]
            delta = epa_post - epa_pre
            print(f"    {m['event_code']:8s} {elim}  pre={epa_pre:6.1f}  post={epa_post:6.1f}  chg={delta:+5.1f}  WP={wp:.0%}")


def compare_teams(engine, team_a, team_b):
    a = engine.get_team(team_a)
    b = engine.get_team(team_b)
    print(f"\n{'='*60}")
    print(f"  Head-to-Head:  Team {team_a}  vs  Team {team_b}")
    print(f"{'='*60}")
    wp = predict_win_prob(engine, a.mean[0], b.mean[0])
    rev = predict_win_prob(engine, b.mean[0], a.mean[0])
    print(f"  Win probability:")
    print(f"    Team {team_a}: {wp:.1%}")
    print(f"    Team {team_b}: {rev:.1%}")
    print(f"\n  EPA comparison:")
    print(f"    {'':>15s}  {f'Team {team_a}':>10s}  {f'Team {team_b}':>10s}  {'Diff':>8s}")
    print(f"    {'-'*45}")
    print(f"    {'Total EPA':>15s}  {a.mean[0]:>10.1f}  {b.mean[0]:>10.1f}  {a.mean[0]-b.mean[0]:>+8.1f}")
    print(f"    {'Auto EPA':>15s}  {a.mean[1]:>10.1f}  {b.mean[1]:>10.1f}  {a.mean[1]-b.mean[1]:>+8.1f}")
    print(f"    {'Teleop EPA':>15s}  {a.mean[2]:>10.1f}  {b.mean[2]:>10.1f}  {a.mean[2]-b.mean[2]:>+8.1f}")
    print(f"    {'Endgame EPA':>15s}  {a.mean[3]:>10.1f}  {b.mean[3]:>10.1f}  {a.mean[3]-b.mean[3]:>+8.1f}")


def leaderboard(engine, meta, top_n=15):
    storage_local = SQLiteStorage(DB_PATH, "2025")
    season_teams = storage_local.load_season_teams("2025")
    totals = [(t, sn.mean[0]) for t, sn in engine.epas.items()]
    totals.sort(key=lambda x: -x[1])
    print(f"\n{'='*65}")
    print(f"  Top {top_n} Teams  (raw mean={np.mean([t[1] for t in totals]):.1f}, "
          f"sd={np.std([t[1] for t in totals]):.1f}, n={len(totals)})")
    print(f"{'='*65}")
    print(f"  {'Rank':>4s}  {'Team':>6s}  {'Raw EPA':>8s}  {'norm_epa':>8s}  {'Auto':>7s}  {'Teleop':>7s}  {'Endgame':>7s}  {'Matches':>7s}")
    print(f"  {'-'*60}")
    for rank, (team, _) in enumerate(totals[:top_n], 1):
        sn = engine.get_team(team)
        norm = season_teams.get(team, {}).get("norm_epa", 0)
        print(f"  {rank:>4d}  {team:>6d}  {sn.mean[0]:>8.1f}  {norm:>8.1f}  {sn.mean[1]:>7.1f}  {sn.mean[2]:>7.1f}  {sn.mean[3]:>7.1f}  {engine.counts.get(team,0):>7d}")

    print(f"\n  Bottom 5:")
    for team, _ in totals[-5:]:
        sn = engine.get_team(team)
        print(f"    Team {team:>6d}:  total={sn.mean[0]:>7.1f}")


def main():
    storage, meta = safe_load()

    if not meta:
        print("No cached data found. Running pipeline (first time, this takes ~2min)...")
        storage, meta, engine = run_pipeline()
    else:
        from backend.src.core.config import get_season_config
        from backend.src.services.epa_service import EPAEngine
        config = get_season_config("2025")
        engine = EPAEngine(config=config)
        engine.score_sd = meta["score_sd"]
        teams = storage.load_all_teams()
        for team, params in teams.items():
            engine.set_team_state(
                team, params["mean"], params["var"],
                params["skew"], params["n"], params["count"],
            )
        print(f"Loaded {len(teams)} teams from cache.")

    print(f"Calibrated score_sd={engine.score_sd:.1f}, score_mean={meta['score_mean']:.1f}")
    print(f"Commands:  team <number>  |  compare <a> <b>  |  top [N]  |  search <term>  |  quit")

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()

        if cmd in ("q", "quit", "exit"):
            break

        elif cmd == "team":
            if len(parts) < 2:
                print("Usage: team <number>")
                continue
            try:
                t = int(parts[1])
            except ValueError:
                print(f"Invalid team number: {parts[1]}")
                continue
            if t not in engine.epas:
                print(f"Team {t} not found in training data.")
                continue
            show_team(storage, engine, meta, t)

        elif cmd in ("compare", "vs", "cmp"):
            if len(parts) < 3:
                print("Usage: compare <team_a> <team_b>")
                continue
            try:
                a, b = int(parts[1]), int(parts[2])
            except ValueError:
                print("Invalid team numbers.")
                continue
            if a not in engine.epas:
                print(f"Team {a} not found."); continue
            if b not in engine.epas:
                print(f"Team {b} not found."); continue
            compare_teams(engine, a, b)

        elif cmd in ("top", "leaderboard"):
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 15
            leaderboard(engine, meta, min(n, len(engine.epas)))

        elif cmd == "search":
            term = parts[1].lower() if len(parts) > 1 else ""
            if not term:
                print("Usage: search <number>")
                continue
            matches = [t for t in engine.epas if term in str(t)]
            if not matches:
                print(f"No teams matching '{term}'")
            else:
                print(f"Teams matching '{term}': {len(matches)} found")
                for t in sorted(matches)[:20]:
                    sn = engine.get_team(t)
                    print(f"  {t:>6d}: total={sn.mean[0]:>7.1f}  matches={engine.counts.get(t,0)}")
                if len(matches) > 20:
                    print(f"  ... and {len(matches)-20} more")

        elif cmd in ("stats", "dist"):
            totals = [sn.mean[0] for sn in engine.epas.values()]
            print(f"\nEPA Distribution:")
            print(f"  Count:   {len(totals)}")
            print(f"  Mean:    {np.mean(totals):.1f}")
            print(f"  StdDev:  {np.std(totals):.1f}")
            print(f"  Median:  {np.median(totals):.1f}")
            print(f"  Min:     {min(totals):.1f}")
            print(f"  Max:     {max(totals):.1f}")
            pcts = [10, 25, 75, 90]
            for p in pcts:
                print(f"  P{p}:     {np.percentile(totals, p):.1f}")

        elif cmd == "help":
            print("Commands:")
            print("  team <num>         Show EPA breakdown for a team")
            print("  compare <a> <b>    Head-to-head comparison")
            print("  top [N]            Leaderboard (default 15)")
            print("  search <term>      Find teams by number")
            print("  stats              Distribution statistics")
            print("  quit               Exit")

        else:
            print(f"Unknown command: {cmd}  (try 'help')")


if __name__ == "__main__":
    main()
