import urllib.request, json

# Test clusters endpoint
req = urllib.request.Request("http://127.0.0.1:8000/v1/clusters?season=2025&n_clusters=5")
resp = json.loads(urllib.request.urlopen(req).read())

print(f"Season: {resp['season']}")
print(f"n_clusters: {resp['n_clusters']}")
print(f"Dimensions: {resp['dimensions']}")
print(f"Teams classified: {len(resp['teams'])}")
print()

for c in resp['clusters']:
    print(f"Cluster {c['id']} ({c['label']}): {c['size']} teams, total EPA mean={c.get('total_points_mean')}")
    print(f"  Center: {c['center']}")
    print(f"  Top teams: {c['top_teams'][:5]}")
    print()

# Test team playstyle endpoint
team = next(iter(resp['teams'].keys()))
req2 = urllib.request.Request(f"http://127.0.0.1:8000/v1/team/{team}/playstyle?season=2025")
resp2 = json.loads(urllib.request.urlopen(req2).read())
print(f"\nTeam {team} playstyle:")
print(json.dumps(resp2, indent=2))
