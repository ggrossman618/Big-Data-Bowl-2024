# Imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

import mappings

# Setting Options 
pd.set_option('display.max_columns', None)

# Defining reusable functions
def getSeconds(time):
    m, s = time.split(':')
    return int(m) * 60 + int(s)

# Starting data manipulation
df = pd.read_csv('./data/plays.csv')
tracking_data = pd.read_csv('./data/tracking_week_1.csv')

df = df.drop(['ballCarrierDisplayName', 'playDescription', 'passResult', 'passLength', 'foulName1', 'foulName2', "foulNFLId1", 'foulNFLId2', 'penaltyYards', 'playNullifiedByPenalty', 'prePenaltyPlayResult', 'expectedPointsAdded', 'homeTeamWinProbabilityAdded', 'visitorTeamWinProbilityAdded'], axis=1)

df['offenseFormation'] = df['offenseFormation'].map(mappings.offensiveFormationsMappings)
df['possessionTeam'] = df['possessionTeam'].map(mappings.teamMappings)
df['defensiveTeam'] = df['defensiveTeam'].map(mappings.teamMappings)
df['yardlineSide'] = df['yardlineSide'].map(mappings.teamMappings)
df['gameClock'] = df['gameClock'].apply(getSeconds)

# new created columns
df['pre_snap_win_prob_diff'] = (df['preSnapHomeTeamWinProbability'] - df['preSnapVisitorTeamWinProbability']).astype(float)
df['score_margin'] = (df['preSnapHomeScore'] - df['preSnapVisitorScore']).astype(float)
df['relative_field_position'] = df['absoluteYardlineNumber'] - 50  # Assuming 50-yard line is midfield
df['in_red_zone'] = (df['absoluteYardlineNumber'] <= 20).astype(int)
df['time_left_in_quarter'] = (900 - df['gameClock']) / 900  # Assuming each quarter is 15 minutes
df['time_pressure'] = (df['gameClock'] < 120).astype(int)  # Assuming 2 minutes left in the game
df['avg_yards_to_go_per_quarter'] = df.groupby('quarter')['yardsToGo'].transform('mean').astype(float)
df['field_position_ratio'] = df['absoluteYardlineNumber'] / 100  # Assuming 100 yards per field length
df['situation_complexity'] = df['defendersInTheBox'] * df['yardsToGo']
df['yards_to_go_trend'] = df['yardsToGo'].diff().fillna(0).astype(float)
df['field_position_change'] = df['absoluteYardlineNumber'].diff().fillna(0).astype(float)
df['time_impact_yards'] = df['gameClock'] * df['yardsToGo']
df['defense_pressure_index'] = df['defendersInTheBox'] * df.groupby('defensiveTeam')['yardsToGo'].transform('mean')
df['defensive_pressure_trend'] = df['defendersInTheBox'].diff().fillna(0).astype(float)
df['relative_field_position_change'] = df['absoluteYardlineNumber'] - 50  # Assuming 50-yard line is midfield
df['relative_field_position_change'] = df['relative_field_position_change'].diff().fillna(0).astype(float)
df['down_distance_complexity_index'] = (df['down'] * df['yardsToGo']).astype(float)
df['formation_diversity_index'] = df.groupby('possessionTeam')['offenseFormation'].transform(lambda x: len(set(x))).astype(float)

# Average distance of defenders from the line of scrimmage before the play starts
#df['avgDefenderLOS'] = df.apply(lambda row: tracking_data[(tracking_data['gameId'] == row['gameId']) & (tracking_data['playId'] == row['playId']) & (tracking_data['event'] == 'ball_snap') & (tracking_data['club'] != row['possessionTeam'])]['x'].mean(), axis=1)

# Maximum speed of defenders before the play starts
#df['maxDefenderSpeed'] = df.apply(lambda row: tracking_data[(tracking_data['gameId'] == row['gameId']) & (tracking_data['playId'] == row['playId']) & (tracking_data['event'] == 'ball_snap') & (tracking_data['club'] != row['possessionTeam'])]['s'].max(), axis=1)

# Average distance of defenders from the ball carrier's starting position
#df['avgDefenderBallCarrierDist'] = df.apply(lambda row: tracking_data[(tracking_data['gameId'] == row['gameId']) & (tracking_data['playId'] == row['playId']) & (tracking_data['event'] == 'ball_snap') & (tracking_data['club'] != row['possessionTeam'])]['x'].mean(), axis=1)

# Number of defenders in close proximity to the ball carrier's starting position
#df['closeDefendersBallCarrier'] = df.apply(lambda row: tracking_data[(tracking_data['gameId'] == row['gameId']) & (tracking_data['playId'] == row['playId']) & (tracking_data['event'] == 'ball_snap') & (tracking_data['club'] != row['possessionTeam']) & (abs(tracking_data['x'] - row['ballCarrierStartingX']) < threshold)]['nflId'].nunique(), axis=1)


df = df.dropna()



# ********** Starting Model Work **********


y = df['playResult']
X = df.drop(['playResult'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

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
