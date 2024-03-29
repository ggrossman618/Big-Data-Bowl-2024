import pandas as pd
import numpy as np
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

# # Machine learning model for predicting offensive formations

teamFormPredictability = {} # Rank teams' predictability based on model accuracy

for team, value in teamMappings.items():
    print('*------------------------------------*')
    print(team)
    print('*-------------------------------------*')
    
    # Setting up data for model
    teamName = team
    team = teamMappings[team].copy()
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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Print accuracy of model
    print("The model's accuracy is: ", np.round(accuracy_score(y_test, y_pred, sample_weight=None) * 100, 4), '%')
    print('')
    
    # Transforming offensive formations back from integers to strings
    testVals = [next((k for k, v in mappings.offensiveFormationsMappings.items() if v == i), i) for i in y_test.values.tolist()]
    predsVals = [next((k for k, v in mappings.offensiveFormationsMappings.items() if v == i), i) for i in y_pred.tolist()]

    comparisonDf = pd.DataFrame({
        'Actual': testVals,
        'Predicted': predsVals
    })
    print('Actual vs Predicted Offensive Formations:')
    print(comparisonDf)
    print('')

    # Print the predicted class and the corresponding probabilities, changing plays from 0-5 to string names
    play = 0
    for i in range(len(y_pred)):  
        formationName = next((k for k, v in mappings.offensiveFormationsMappings.items() if v == y_pred[i]), None)
        classes = model.classes_
        print(f"Play { play }")
        print(f"Predicted Formation: { formationName }")
        print(f"Actual Formation: { comparisonDf['Actual'][i] }")
        print("Probabilities")
        
        x = 0
        for class_name, proba in zip(classes, y_pred_proba):
            class_name_string = next((k for k, v in mappings.offensiveFormationsMappings.items() if v == class_name), None)
            print(f"- { class_name_string }: { np.round(y_pred_proba[i][x], 4) * 100 }% ")
            x += 1
        
        play += 1
        print('')
    
    print('')
    print("Classification Report:")
    print(classification_report(y_test , y_pred))
    
    teamFormPredictability[teamName] = round(accuracy_score(y_test, y_pred, sample_weight=None), 2)
    

print('*------------------------------------*')
print('')

# Sorting team predictibility dictionary, then converting to dataframe
teamFormPredictability = dict(sorted(teamFormPredictability.items(), key=lambda item: item[1]))
teamFormPredictability = {'Team': list(teamFormPredictability.keys()), 'Predictability': list(teamFormPredictability.values())}
teamFormPredictability = pd.DataFrame(teamFormPredictability)
print('Team Predictability Rankings:')
print(teamFormPredictability)


# Play Type
print('')
print('*** PLAY TYPE ***')
print('')

# Running analytics per team and creating graphs
#Offensive Play Types
for key, value in teamMappings.items():
    if not teamMappings[key].empty:
        # set running plays to 0 and passing plays to 1
        teamMappings[key]['playTypeGraph'] = teamMappings[key]['passResult'].apply(lambda x: 0 if pd.isna(x) else 1)
        teamMappings[key]['playTypeGraph'] = teamMappings[key]['playTypeGraph'].apply(lambda x: 'Run' if x == 0 else 'Pass')

        offensePlayPercentages = teamMappings[key]['playTypeGraph'].value_counts(normalize=True) * 100
        plt.figure(figsize=(8, 6))
        offensePlayPercentages.plot(kind='bar', color='skyblue')
        plt.ylim(0, 100)  # Set the y-axis limit to 100
        plt.title(f"{key}'s Retaliation Play Type Percentage")
        plt.xlabel('Offensive Play')
        plt.ylabel('Percentage')
        plt.xticks(rotation=0)  # Rotate x-labels if needed
        plt.tight_layout()
        plt.show()

teamPlayPredictability = {} # Rank teams' predictability based on model accuracy

for team, value in teamMappings.items():
    print('*------------------------------------*')
    print(team)
    print('*-------------------------------------*')
    
    # Setting up data for model
    teamName = team
    team = teamMappings[team].copy()
    team['possessionTeam'] = team['possessionTeam'].map(mappings.teamMappings)
    team['defensiveTeam'] = team['defensiveTeam'].map(mappings.teamMappings)
    team['yardlineSide'] = team['yardlineSide'].map(mappings.teamMappings)
    team['offenseFormation'] = team['offenseFormation'].map(mappings.offensiveFormationsMappings)
    team['gameClock'] = team['gameClock'].apply(getSeconds)
    
    # set running plays to 0 and passing plays to 1
    team['playType'] = team['passResult'].apply(lambda x: 0 if pd.isna(x) else 1)
    
    team = team.drop(['playResult', 'defendersInTheBox', 'ballCarrierDisplayName', 'expectedPoints', 'ballCarrierId', 'playDescription', 'passResult', 'passLength', 'foulName1', 'foulName2', "foulNFLId1", 'foulNFLId2', 'penaltyYards', 'playNullifiedByPenalty', 'prePenaltyPlayResult', 'expectedPointsAdded', 'homeTeamWinProbabilityAdded', 'visitorTeamWinProbilityAdded', 'playTypeGraph'], axis=1)
    team = team.dropna()

    y = team['playType']
    X = team.drop(['playType'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) # use test size 25 for individual teams
    
    # Train the model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Print accuracy of model
    print("The model's accuracy is: ", np.round(accuracy_score(y_test, y_pred, sample_weight=None) * 100, 4), '%')
    print('')

    comparisonDf = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })

    comparisonDf['Actual'] = comparisonDf['Actual'].apply(lambda x: 'Run' if x == 0 else 'Pass')
    comparisonDf['Predicted'] = comparisonDf['Predicted'].apply(lambda x: 'Run' if x == 0 else 'Pass')

    print(comparisonDf)

    teamPlayPredictability[teamName] = round(accuracy_score(y_test, y_pred, sample_weight=None), 2)

# Sorting team predictibility dictionary, then converting to dataframe
teamPlayPredictability = dict(sorted(teamPlayPredictability.items(), key=lambda item: item[1]))
teamPlayPredictability = {'Team': list(teamPlayPredictability.keys()), 'Predictability': list(teamPlayPredictability.values())}
teamPlayPredictability = pd.DataFrame(teamPlayPredictability)
print('')
print('Team Play Predictability Rankings:')
print(teamPlayPredictability)
