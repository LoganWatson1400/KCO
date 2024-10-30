import os
import sys
import pandas as pd
import numpy as np
import random
from Roll_Inventory_Optimizer_Scoring import officialScorer
# import Roll_Inventory_Optimizer_Scoring as scoring

week = 0
weeks = [
    '2024-09-06 Week 1',
    '2024-09-06 Week 2',
    '2024-09-06 Week 3'
]

bestSchedule = 'HackathonPackageV1/BestSchedule.json'
root = 'HackathonPackageV1/DataCache/OptimizerSituations'
outRoot = 'HackathonPackageV1/PredDataCache/OptimizerSituations'
staticPath = f'HackathonPackageV1/DataCache/OptimizerSituations/ {weeks[week]}/planningSchedule.json'
            #   'HackathonPackageV1/DataCache/OptimizerSituations/2024-09-06 Week 1/planningSchedule.json'
outPath = f'HackathonPackageV1/PredDataCache/OptimizerSituations/{weeks[week]}/planningSchedule.json'

df = pd.read_json('HackathonPackageV1/DataCache/OptimizerSituations/2024-09-06 Week 1/planningSchedule.json')
### Catagorical Data ###
PUnits = df['ProductionUnit'].unique()
PIds = df['Prod_Id'].unique()

### Float Values ###
ForeStartMIN = df['ForecastStartTime'].min()
ForeStartMAX = df['ForecastStartTime'].max()

ForeEndMIN = df['ForecastEndTime'].min()
ForeEndMAX = df['ForecastEndTime'].max()
 




### randomizer ###TODO

### Randomization function
def randomize_value(df, column):
    if column in ['ProductionUnit', 'Prod_Id']:
        return np.random.choice(df[column].unique())
    elif column == 'ForecastStartTime':
        return np.random.uniform(ForeStartMIN, ForeStartMAX)
    elif column == 'ForecastEndTime':
        return np.random.uniform(ForeEndMIN, ForeEndMAX)

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


(best, z) = officialScorer(root, weeks[week])
iterations = 0
patients = 0
# blockPrint() ##########################################################
while iterations < 4 and patients < 5:
    iterations += 1
    patients += 1
    

    new_PUnits = random.choices(PUnits, k=len(df['ProductionUnit']))
    # new_PIds = random.choices(PIds, k=len(df['Prod_Id']))

    df['ProductionUnit'] = new_PUnits
    # df['Prod_Id'] = new_PIds

    #TODO random start and end time

    df.to_json(outPath, indent=4)
    (loss, z) = officialScorer(outRoot, weeks[week]) #week is default #TODO can use breakdown to optimize
    # print(f'loss: {loss}: breakdown: {breakdown}')
    if loss > best:
        best = loss
        df.to_json(bestSchedule, indent=4)
        df.to_csv('tag_value_pairs.csv')
        patients = 0
    
    print('\n\n')
# enablePrsint() ########################################################

print(f'Best Score achived: {best}')
temp = pd.read_json(bestSchedule)
temp.to_json(outPath, indent=4)
officialScorer(outRoot, weeks[week])


