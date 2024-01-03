import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings # for getting rid of warnings
from sklearn.exceptions import UndefinedMetricWarning # for getting rid of warnings

import mappings

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Defining reusable functions
def getSeconds(time):
    m, s = time.split(':')
    return int(m) * 60 + int(s)

# Read in data from CSV files
plays = pd.read_csv('./data/plays.csv')

# Sort plays within each game
sortedPlays = plays.sort_values(by=['gameId', 'playId'])

# Identify plays with negative or zero playResult
negativePlays = sortedPlays[(sortedPlays['playResult'] <= 2)]

# Retrieve the indexes of the identified plays
negativePlaysIndexes = negativePlays.index.tolist()

# Get the next play after negative plays
retaliationPlaysIndexes = [index + 1 for index in negativePlaysIndexes if (index + 1) in sortedPlays.index]

# Get the retaliation plays
retaliationPlays = sortedPlays.loc[retaliationPlaysIndexes]

# Filter retaliationPlays to only contain first and second downs
retaliationPlaysFiltered = retaliationPlays[(retaliationPlays['down'] < 3)]

# Map retaliation plays to teams
teamMappings = {
    'ARI': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'ARI'],
    'ATL': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'ATL'],
    'BAL': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'BAL'],
    'BUF': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'BUF'],
    'CAR': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'CAR'],
    'CHI': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'CHI'],
    'CIN': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'CIN'],
    'CLE': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'CLE'],
    'DAL': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'DAL'],
    'DEN': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'DEN'],
    'DET': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'DET'],
    'GB': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'GB'],
    'HOU': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'HOU'],
    'IND': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'IND'],
    'JAX': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'JAX'],
    'KC': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'KC'],
    'LAC': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'LAC'],
    'LAR': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'LA'],
    'LV': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'LV'],
    'MIA': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'MIA'],
    'MIN': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'MIN'],
    'NE': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'NE'],
    'NO': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'NO'],
    'NYG': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'NYG'],
    'NYJ': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'NYJ'],
    'PHI': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'PHI'],
    'PIT': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'PIT'],
    'SEA': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'SEA'],
    'SF': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'SF'],
    'TB': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'TB'],
    'TEN': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'TEN'],
    'WAS': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'WAS']
}

# Running analytics per team and creating graphs

#Offensive Formations
for key, value in teamMappings.items():
    if not teamMappings[key].empty:
        offenseFormationPercentages = teamMappings[key]['offenseFormation'].value_counts(normalize=True) * 100
        plt.figure(figsize=(8, 6))
        offenseFormationPercentages.plot(kind='bar', color='skyblue')
        plt.ylim(0, 100)  # Set the y-axis limit to 100
        plt.title(f"{key}'s Retaliation Play Formation Percentage")
        plt.xlabel('Offensive Formation')
        plt.ylabel('Percentage')
        plt.xticks(rotation=0)  # Rotate x-labels if needed
        plt.tight_layout()
        plt.show()

# Machine Learning Stuff


# Rank teams' predictability 

teamPredictability = {}

for team, value in teamMappings.items():
    print('*------------------------------------*')
    print(team)
    print('*-------------------------------------*')
    teamName = team
    team = teamMappings[team]
    team['possessionTeam'] = team['possessionTeam'].map(mappings.teamMappings)
    team['defensiveTeam'] = team['defensiveTeam'].map(mappings.teamMappings)
    team['yardlineSide'] = team['yardlineSide'].map(mappings.teamMappings)
    team['offenseFormation'] = team['offenseFormation'].map(mappings.offensiveFormationsMappings)
    team['gameClock'] = team['gameClock'].apply(getSeconds)
    team = team.drop(['playResult', 'defendersInTheBox', 'ballCarrierDisplayName', 'expectedPoints', 'ballCarrierId', 'playDescription', 'passResult', 'passLength', 'foulName1', 'foulName2', "foulNFLId1", 'foulNFLId2', 'penaltyYards', 'playNullifiedByPenalty', 'prePenaltyPlayResult', 'expectedPointsAdded', 'homeTeamWinProbabilityAdded', 'visitorTeamWinProbilityAdded'], axis=1)
    team = team.dropna()
    
    y = team['offenseFormation']
    X = team.drop(['offenseFormation'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) # use test size 25 for individual teams
    
    # Train the model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Print the predicted class and the corresponding probabilities
    # for i in range(len(y_pred)):
    #     print(f"Predicted class: {y_pred[i]}, probabilities: {y_pred_proba[i]}")

    teamPredictability[teamName] = round(accuracy_score(y_test, y_pred, sample_weight=None), 2)
    print("The model's accuracy is: ", accuracy_score(y_test, y_pred, sample_weight=None))
    print("Classification Report:")
    print(classification_report(y_test , y_pred))
    
    # Transforming offensive formations back from integers to strings
    testVals = [next((k for k, v in mappings.offensiveFormationsMappings.items() if v == i), i) for i in y_test.values.tolist()]
    predsVals = [next((k for k, v in mappings.offensiveFormationsMappings.items() if v == i), i) for i in y_pred.tolist()]

    comparisonDf = pd.DataFrame({
        'Actual': testVals,
        'Predicted': predsVals
    })
    print('Actual vs Predicted Offensive Formations:')
    print(comparisonDf)

print('*------------------------------------*')
print('')

# Sorting team predictibility dictionary, then converting to dataframe
teamPredictability = dict(sorted(teamPredictability.items(), key=lambda item: item[1]))
teamPredictability = {'Team': list(teamPredictability.keys()), 'Predictability': list(teamPredictability.values())}
teamPredictability = pd.DataFrame(teamPredictability)
print('Team Predictability Rankings:')
print(teamPredictability)
