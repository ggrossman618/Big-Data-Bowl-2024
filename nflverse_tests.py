# Imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import mappings

pd.set_option('display.max_columns', None)


# Read in data
nflverse_plays = pd.read_csv('./data/play_by_play_2022.csv')
bdb_plays = pd.read_csv('./data/plays.csv')


# Defining reusable functions
def getSeconds(time):
    m, s = time.split(':')
    return int(m) * 60 + int(s)


# Filtering DataFrames
nflverse_columns = [
    'play_id',
    'old_game_id',
    'home_team',
    # 'away_team',
    # 'posteam',
    'posteam_type',
    # 'defteam',
    'half_seconds_remaining',
    'ydstogo',
    'shotgun',
    'no_huddle',
    'posteam_score',
    'defteam_score',
    'score_differential',
    'no_score_prob',
    'opp_safety_prob',
    'opp_td_prob',
    'fg_prob',
    'safety_prob',
    'td_prob',
    'ep',
    'wp',
    'cp',
    'cpoe',
    'down',
    # 'weather',
    'play_clock',
    'drive_play_count',
    'drive_time_of_possession',
    'drive_first_downs',
    'drive_inside20',
    'ydstogo',
    'yardline_100',
    'yards_gained'
]

bdb_plays = bdb_plays.drop(['ballCarrierDisplayName', 'playDescription', 'passResult', 'passLength', 'foulName1', 'foulName2', "foulNFLId1", 'foulNFLId2', 'penaltyYards', 'playNullifiedByPenalty', 'prePenaltyPlayResult', 'expectedPointsAdded', 'homeTeamWinProbabilityAdded', 'visitorTeamWinProbilityAdded'], axis=1)
nflverse_plays = nflverse_plays[nflverse_columns]


# Merging the two data frame columns 
nflverseSortedPlays = nflverse_plays.sort_values(by=['old_game_id', 'play_id'])
bdbSortedPlays = bdb_plays.sort_values(by=['gameId', 'playId'])

merged_df = pd.merge(bdbSortedPlays, nflverseSortedPlays, left_on=['gameId', 'playId'], right_on=['old_game_id', 'play_id'], how='left')


# Adjusting rows to correct format
merged_df['offenseFormation'] = merged_df['offenseFormation'].map(mappings.offensiveFormationsMappings)
merged_df['possessionTeam'] = merged_df['possessionTeam'].map(mappings.teamMappings)
merged_df['defensiveTeam'] = merged_df['defensiveTeam'].map(mappings.teamMappings)
merged_df['yardlineSide'] = merged_df['yardlineSide'].map(mappings.teamMappings)
merged_df['home_team'] = merged_df['home_team'].map(mappings.teamMappings)
#merged_df['away_team'] = merged_df['away_team'].map(mappings.teamMappings)
#merged_df['posteam'] = merged_df['posteam'].map(mappings.teamMappings)
merged_df['gameClock'] = merged_df['gameClock'].apply(getSeconds)
merged_df['posteam_type'] = merged_df['posteam_type'].map(mappings.posessionTeamType)
merged_df['drive_time_of_possession'] = merged_df['drive_time_of_possession'].apply(getSeconds)


# Adding synthetic columns
merged_df['field_to_endzone'] = 100 - merged_df['yardline_100']
merged_df['yards_gained_previous_play'] = merged_df.groupby('old_game_id')['playResult'].shift().fillna(0)



merged_df = merged_df.dropna()
merged_df.head()


# Seporate data for model
y = merged_df['playResult']
X = merged_df.drop(['yards_gained', 'playResult'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Model analytics
print("this model's play result predictions are off by approximately: ", mean_absolute_error(y_test, y_pred), ' yards')
data = {'True Values': y_test, 'Predicted Values': y_pred}

# Calculate the errors between predictions and actual test data
errors = y_pred - y_test  # Difference between predicted and actual values

# Calculate the standard deviation of errors using np.std()
print('the standard deviations between predictions and test data is: ', np.std(errors))

# Create a DataFrame to store feature importances with column names
importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})

# Sort the DataFrame by importance values
importances = importances.sort_values(by='Importance', ascending=False)

print(importances)

data = {'True Values': y_test, 'Predicted Values': y_pred}
dfTest = pd.DataFrame(data)
dfTest





'''
Potentially useful rows
- play_id
- old_game_id
- home_team: Team playing at home
- away_team: Team playing away
- posteam: Team with possession
- posteam_type: Type of team with possession (offense/defense)
- defteam: Team on defense
- half_seconds_remaining: Seconds remaining in the half
- ydstogo: Yards to go for a first down
- shotgun: Indicator for a shotgun formation
- no_huddle: Indicator for no-huddle offense
- posteam_score: Score of the team with possession
- defteam_score: Score of the defending team
- score_differential: Difference in scores between teams
- no_score_prob: Probability of no scoring on the next play
- opp_safety_prob: Probability of the opponent scoring a safety
- opp_td_prob: Probability of the opponent scoring a touchdown
- fg_prob: Probability of scoring a field goal
- safety_prob: Probability of scoring a safety
- td_prob: Probability of scoring a touchdown
- ep: Expected points for the possession
- wp: Win probability for the possessing team
- cp: Completion percentage
- cpoe: Completion percentage over expectation
- weather: Weather conditions during the game
- play_clock: Duration of the play clock
- drive_play_count: Number of plays in the drive
- drive_time_of_possession: Time of possession for the drive
- drive_first_downs: Number of first downs in the drive
- drive_inside20: Indicator for being inside the 20-yard line in the drive


useful functions
.map(mappings.offensiveFormationsMappings)

# To read columns
columns_list = merged_df.columns.tolist()
for column_name in columns_list:
    print(column_name)
'''