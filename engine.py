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