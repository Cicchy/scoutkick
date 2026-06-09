import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from backend.src.data.cleaner import CleanerRegistry
from backend.src.data.read_ftcscout import get_matches
from backend.src.storage.sqlite_storage import SQLiteStorage

DB_PATH = "cache/epa_data.db"

def load_actual_outcomes(season_id: str) -> dict:
    cleaner = CleanerRegistry.get_cleaner(season_id)
    raw = get_matches(cleaner, cache=True)
    lookup = {}
    for m in raw:
        ec = m["event"]
        mid = str(m["match_id"])
        rs = m["red_scores"].get("totalPointsNp", 0)
        bs = m["blue_scores"].get("totalPointsNp", 0)
        if rs == bs:
            continue
        lookup[(ec, mid)] = {
            "red_score": rs,
            "blue_score": bs,
            "red_won": rs > bs,
            "red_teams": m["red_teams"],
            "blue_teams": m["blue_teams"],
        }
    return lookup


def validate_season(season_id: str):
    print(f"\n{'='*60}")
    print(f"  Validating Season {season_id}")
    print(f"{'='*60}")

    outcomes = load_actual_outcomes(season_id)
    storage = SQLiteStorage(DB_PATH, season_id)
    all_teams = storage.load_all_teams()

    if not all_teams:
        print("  No teams found in DB. Skipping.")
        return

    preds = []
    actuals = []
    missed = 0
    total = 0

    for team in all_teams:
        matches = storage.load_team_matches(team)
        for m in matches:
            total += 1
            key = (m["event_code"], m["match_id"])
            outcome = outcomes.get(key)
            if outcome is None:
                missed += 1
                continue
            # Determine if this team's alliance actually won
            team_won = (
                (team in outcome["red_teams"] and outcome["red_won"]) or
                (team in outcome["blue_teams"] and not outcome["red_won"])
            )
            preds.append(m["win_prob"])
            actuals.append(1.0 if team_won else 0.0)

    preds = np.array(preds)
    actuals = np.array(actuals)

    if len(preds) == 0:
        print("  No matched predictions. Skipping.")
        return

    n = len(preds)
    brier = np.mean((preds - actuals) ** 2)
    log_loss = -np.mean(actuals * np.log(np.clip(preds, 1e-15, 1)) +
                        (1 - actuals) * np.log(np.clip(1 - preds, 1e-15, 1)))
    predicted_wins = (preds > 0.5).astype(float)
    accuracy = np.mean(predicted_wins == actuals)
    avg_pred = np.mean(preds)
    base_rate = np.mean(actuals)

    print(f"  Matches:         {n}")
    if missed:
        print(f"  Skipped (tie):   {missed}")
    print(f"  Avg prediction:  {avg_pred:.4f}")
    print(f"  Base win rate:   {base_rate:.4f}")
    print(f"  Accuracy:        {accuracy:.4f}")
    print(f"  Brier score:     {brier:.4f}  (0=perfect, 0.25=naive)")
    print(f"  Log loss:        {log_loss:.4f}  (0=perfect)")

    # Calibration curve
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_acc = []
    bin_counts = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (preds >= lo) & (preds < hi)
        count = mask.sum()
        bin_counts.append(count)
        if count > 0:
            bin_acc.append(actuals[mask].mean())
        else:
            bin_acc.append(np.nan)

    print(f"\n  Calibration:")
    print(f"  {'Bin':>10s}  {'Count':>6s}  {'Win Rate':>9s}  {'Expected':>9s}")
    for i in range(len(bins) - 1):
        exp = (bins[i] + bins[i + 1]) / 2
        actual_val = bin_acc[i]
        if not np.isnan(actual_val):
            bar = "#" * int(actual_val * 20)
        else:
            bar = "-"
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}  {bin_counts[i]:>6d}  {actual_val if not np.isnan(actual_val) else float('nan'):>9.4f}  {exp:>9.4f}  {bar}")

    ece = np.nansum(np.array(bin_counts) * np.abs(np.array(bin_acc) - bin_centers)) / max(n, 1)
    print(f"  ECE:             {ece:.4f}  (Expected Calibration Error)")

    return {
        "season": season_id,
        "n": n,
        "accuracy": accuracy,
        "brier": brier,
        "log_loss": log_loss,
        "ece": ece,
        "avg_pred": avg_pred,
        "base_rate": base_rate,
    }


if __name__ == "__main__":
    results = []
    for s in ["2025", "2024", "2023", "2022", "2021", "2020", "2019"]:
        r = validate_season(s)
        if r:
            results.append(r)

    if results:
        print(f"\n{'='*60}")
        print(f"  Summary")
        print(f"{'='*60}")
        print(f"  {'Season':>6s}  {'Matches':>8s}  {'Accuracy':>9s}  {'Brier':>7s}  {'LogLoss':>8s}  {'ECE':>7s}")
        for r in results:
            print(f"  {r['season']:>6s}  {r['n']:>8d}  {r['accuracy']:>9.4f}  {r['brier']:>7.4f}  {r['log_loss']:>8.4f}  {r['ece']:>7.4f}")
