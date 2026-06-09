import sqlite3, os

db_path = os.path.join(os.getcwd(), "cache", "epa_data.db")
print(f"DB path: {db_path}")
print(f"DB exists: {os.path.exists(db_path)}")
if not os.path.exists(db_path):
    exit()

conn = sqlite3.connect(db_path)
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print(f"Tables: {[t[0] for t in tables]}")
for t in [t[0] for t in tables]:
    cnt = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  {t}: {cnt} rows")
conn.close()
