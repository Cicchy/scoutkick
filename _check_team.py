import sqlite3

conn = sqlite3.connect("cache/epa_data.db")
conn.row_factory = sqlite3.Row

team = 26914
rows = conn.execute(
    "SELECT season, count, norm_epa FROM team_seasons WHERE team = ? ORDER BY season",
    (team,),
).fetchall()

if rows:
    for r in rows:
        print(f"Season {r['season']}: count={r['count']}, norm_epa={r['norm_epa']}")
else:
    print(f"Team {team} NOT FOUND in any season")

print()
rows2 = conn.execute(
    "SELECT season, COUNT(*) as cnt FROM team_seasons GROUP BY season ORDER BY season"
).fetchall()
for r in rows2:
    print(f"Season {r['season']}: {r['cnt']} teams")

conn.close()
