# Original Tests File
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

# Engine.py tests

# Packages
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

# Local files
import mappings

# Global configs
warnings.filterwarnings("ignore")

def createFilteredDataFrame(df, possessionTeam, defensiveTeam, offenseFormation, down, absoluteYardlineNumber, gameClock,passProbability, 
                            onlyVsDefTeam, includeDown, includeRedzone, includeTwoMin, includePassProb, passProbPercentDiff, retaliationPlay):
    filteredDataFrame = None
    
    if retaliationPlay == False:
        filteredDataFrame = df[
            (df['possessionTeam'] == possessionTeam) 
            & (df['defensiveTeam'] == defensiveTeam if onlyVsDefTeam else True)
            & (df['offenseFormation'] == offenseFormation) 
            & (df['down'] == down if includeDown else True)
            & ((df['absoluteYardlineNumber'] <= 20) if (includeRedzone and absoluteYardlineNumber <= 20) else True)
            & ((df['gameClock'].apply(getSeconds) <= 120) if (includeTwoMin and getSeconds(gameClock) <= 120) else True)
            & (((df['passProbability'] <= passProbability + passProbPercentDiff/100) & (df['passProbability'] >= passProbability - passProbPercentDiff/100)) if includePassProb else True)
            & (df['penaltyYards'].isnull()) 

        ]

    else: 
        # Sort plays within each game
        sortedPlays = df.sort_values(by=['gameId', 'playId'])

        # Identify plays with negative or zero playResult
        negativePlays = sortedPlays[(sortedPlays['playResult'] <= 0)]

        # Retrieve the indexes of the identified plays
        negativePlaysIndexes = negativePlays.index.tolist()

        # Get the next play after negative plays
        retaliationPlaysIndexes = [index + 1 for index in negativePlaysIndexes if (index + 1) in sortedPlays.index]

        # Get the retaliation plays
        retaliationPlays = sortedPlays.loc[retaliationPlaysIndexes]

        # Filter retaliationPlays to only contain first and second downs
        filteredDataFrame = retaliationPlays[(retaliationPlays['down'] < 3)]
        
    return filteredDataFrame 


def getSeconds(time):
    m, s = time.split(':')
    return int(m) * 60 + int(s)


def prepareDataForModel(df):
    print(df.head())
    teamMappings = mappings.teamMappings
    offensiveFormationsMappings = mappings.offensiveFormationsMappings

    preparedDf = df[['defendersInTheBox','playResult']]
    preparedDf['offenseFormation'] = df['offenseFormation'].map(offensiveFormationsMappings)

    if 'defensiveTeam' in df:
        preparedDf['defensiveTeam'] = df['defensiveTeam'].map(teamMappings)
    
    if 'down' in df:
        preparedDf['down'] = df['down']

    if 'absoluteYardlineNumber' in df:
        preparedDf['absoluteYardlineNumber'] = df['absoluteYardlineNumber']
    
    if 'gameClock' in df:
        preparedDf['gameClock'] = df['gameClock'].apply(getSeconds)

    if 'passProbability' in df:
        preparedDf['passProbability'] = df['passProbability']

    # Drop NaN values
    preparedDf = preparedDf.dropna()

    y = preparedDf['playResult']
    X = preparedDf.drop('playResult', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    preparedData = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

    return preparedData


def trainModel(data):
    rfModel = RandomForestClassifier()
    rfModel.fit(data['X_train'], data['y_train'])

    return rfModel


def modelTests(model, preparedData):
    print(preparedData['X_train'])
    
    # #print(model.feature_importances_)
    y_pred = model.predict(preparedData['X_test'])

    # # Calculate absolute differences between predictions and actual values
    absolute_diff = abs(preparedData['y_test'] - y_pred)

    # # Count the number of predictions within the 0-3 yard range
    within_range_count = sum((absolute_diff >= 0) & (absolute_diff <= 2))

    # # Calculate the percentage of predictions within the 0-3 yard range
    percentage_within_range = (within_range_count / len(y_pred)) * 100

    #print(f"Percentage of Predictions within 0-3 yards: {percentage_within_range:.2f}%")

    #print(accuracy_score(preparedData['y_test'], y_pred))

    # print(model.feature_importances_)

    accuracy = model.selected.score(preparedData['X_test'], preparedData['y_test'])
    print("Accuracy with selected features:", accuracy)






def main(possessionTeam, defensiveTeam, offenseFormation, down, absoluteYardlineNumber, gameClock, passProbability, 
         onlyVsDefTeam, includeDown, includeRedzone, includeTwoMin, includePassProb, passProbPercentDiff, retaliationPlayOverride):
    games = pd.read_csv('./data/games.csv')
    players = pd.read_csv('./data/players.csv')
    plays = pd.read_csv('./data/plays.csv')
    tackles = pd.read_csv('./data/tackles.csv')

    filteredDf = (createFilteredDataFrame(plays, possessionTeam, defensiveTeam, offenseFormation, down, absoluteYardlineNumber, gameClock, passProbability, 
                                  onlyVsDefTeam, includeDown, includeRedzone, includeTwoMin, includePassProb, passProbPercentDiff, retaliationPlayOverride))

    preparedData = prepareDataForModel(filteredDf)

    model = trainModel(preparedData)

    modelTests(model, preparedData)

    


    

    
    

main(
    'PHI', # Posession team
    'DAL', # Defensive team
    'SHOTGUN', # Offensive formation
    3, # Down 
    20, # AbsoluteYardlineNumber
    '2:00', # gameClock 
    .5, # Pass probability
    False, # Posession team only vs defensive team
    False, # Only use current down in calculation
    False, # Only use plays while in redzone
    False, # Only use plays in last two minutes of quarter
    False, # Only use plays with similar pass probabilities
    .2, # Percent difference in passs probabilities to include
    False # Retaliation play override
)



'''
Will likely need to train multiple models based on numbers of defenders in the box and then compare results
X = plays[['possessionTeam', 'defensiveTeam', 'offenseFormation', 'down', 'yardlineNumber', 'gameClock', 'playResult', 'defendersInTheBox', 'passProbability']]
y = plays['playResult']

Key play data to look at:
- possession team
- defensive team
- offensive formation
- down
- play result (yards gained by offense)
- yardline number
- game clock
- passResult (this includes SACK)
- play result
- absolute yardline number (distance from endzone)
- defenders in the box
- yards to go 
- pass probability

notes:
- I will need to refactor createFilteredDataFrame to always include the columns but have it filter out rows based on user selected arguments
'''

# **** NFLVERSE ORIGINAL FILE

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