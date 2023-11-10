import pandas as pd
import mappings

def createFilteredDataFrame(df, possessionTeam, defensiveTeam, offenseFormation, down, absoluteYardlineNumber, gameClock,passProbability, 
                            onlyVsDefTeam, includeDown, includeRedzone, includeTwoMin, includePassProb, passProbPercentDiff):
    filteredDataFrame = df[
        (df['possessionTeam'] == possessionTeam) 
        & (df['defensiveTeam'] == defensiveTeam if onlyVsDefTeam else True)
        & (df['offenseFormation'] == offenseFormation) 
        & (df['down'] == down if includeDown else True)
        & ((df['absoluteYardlineNumber'] <= 20) if (includeRedzone and absoluteYardlineNumber <= 20) else True)
        & ((df['gameClock'].apply(getSeconds) <= 120) if (includeTwoMin and getSeconds(gameClock) <= 120) else True)
        & (((df['passProbability'] <= passProbability + passProbPercentDiff/100) & (df['passProbability'] >= passProbability - passProbPercentDiff/100)) 
           if includePassProb else True)
    ]

    filteredDataFrame = filteredDataFrame[['possessionTeam', 'offenseFormation', 'defendersInTheBox','playResult']
                                          + (['defensiveTeam'] if onlyVsDefTeam else [])
                                          + (['down'] if includeDown else [])
                                          + (['absoluteYardlineNumber'] if (includeRedzone and absoluteYardlineNumber <= 20) else [])
                                          + (['gameClock'] if (includeTwoMin and getSeconds(gameClock) <= 120) else [])
                                          + (['passProbability'] if includePassProb else [])
                                          ]
    
    return filteredDataFrame 


def prepareDataForModel(df):
    teamMappings = mappings.teamMappings
    offensiveFormationsMappings = mappings.offensiveFormationsMappings

    preparedDf = df[['defendersInTheBox','playResult']]
    df.loc[:, 'possessionTeam'] = df['possessionTeam'].map(teamMappings)
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

    return preparedDf


def getSeconds(time):
    m, s = time.split(':')
    return int(m) * 60 + int(s)


def main(possessionTeam, defensiveTeam, offenseFormation, down, absoluteYardlineNumber, gameClock, passProbability, 
         onlyVsDefTeam, includeDown, includeRedzone, includeTwoMin, includePassProb, passProbPercentDiff):
    games = pd.read_csv('./data/games.csv')
    players = pd.read_csv('./data/players.csv')
    plays = pd.read_csv('./data/plays.csv')
    tackles = pd.read_csv('./data/tackles.csv')

    filteredDf = (createFilteredDataFrame(plays, possessionTeam, defensiveTeam, offenseFormation, down, absoluteYardlineNumber, gameClock, passProbability, 
                                  onlyVsDefTeam, includeDown, includeRedzone, includeTwoMin, includePassProb, passProbPercentDiff))
    
    preparedDfArr = prepareDataForModel(filteredDf)
    print(preparedDfArr.head())
    
    

main('PHI', 'DAL', 'SHOTGUN', 3, 20, '2:00', .5, False, False, False, False, False, .2)



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


'''