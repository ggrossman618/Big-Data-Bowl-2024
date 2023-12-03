import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)

# Read in data from CSV files
plays = pd.read_csv('./data/plays.csv')

# Sort plays within each game
sortedPlays = plays.sort_values(by=['gameId', 'playId'])

# Identify plays with negative or zero playResult
negativePlays = sortedPlays[(sortedPlays['playResult'] <= 0)]

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
    'LAR': retaliationPlaysFiltered[retaliationPlaysFiltered['possessionTeam'] == 'LAR'],
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



''' 
notes:
-for some reason no retaliation plays for rams
'''