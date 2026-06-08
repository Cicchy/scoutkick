from scoutkick_api import ScoutKick
sk = ScoutKick()


team = sk.get_team(32736)

# Predict a match
predict = sk.predict(red=[32753 , 32736], blue=[34143,34148])

# Compare teams
compare = sk.compare(teams=[26914, 32736, 23400, 24599])


print("Team 32736:", team)
print("Match Prediction (Red vs Blue):", predict)
print("Team Comparison:", compare)
