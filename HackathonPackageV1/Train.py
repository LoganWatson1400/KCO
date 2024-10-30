import glob
import os
import sys
import pandas as pd
import numpy as np
import random
from Roll_Inventory_Optimizer_Scoring import officialScorer

week = 0
weeks = [
    '2024-09-06 Week 1',
    '2024-09-06 Week 2',
    '2024-09-06 Week 3'
]

#Static paths
staticPath = f'HackathonPackageV1\\DataCache\\OptimizerSituations\\{weeks[week]}\\planningSchedule.json'
root = 'HackathonPackageV1\\DataCache\\OptimizerSituations'
InitialPaths = glob.glob(root + f'\\{weeks[week]}\\*.json')
temp = {}
for i in InitialPaths:
    temp[os.path.basename(i)] = i
InitialPaths = temp

# print(InitialPaths['initialPOs.json']) #How to get

#Output Paths
outRoot = 'HackathonPackageV1\\PredDataCache\\OptimizerSituations'
outSchedule = f'HackathonPackageV1\\PredDataCache\\OptimizerSituations\\{weeks[week]}\\planningSchedule.json'

#TempPaths
bestSchedule = 'HackathonPackageV1\\BestSchedule.json'

df = pd.read_json(staticPath)
IData = pd.read_json(InitialPaths['initialPOs.json'])

### Catagorical Data ###
PUnits = IData['ProductionUnit'].unique()
PIds = IData['Prod_Id'].unique()

### Float Values ###
ForeStartMIN = IData['ForecastStartTime'].min()
ForeStartMAX = IData['ForecastStartTime'].max()

ForeEndMIN = IData['ForecastEndTime'].min()
ForeEndMAX = IData['ForecastEndTime'].max()
 




### randomizer ###TODO

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


best = -364453
iterations = 0
patients = 0
# blockPrint() ##########################################################
while iterations < 4 and patients < 5:
    iterations += 1
    patients += 1
    

    new_PUnits = random.choices(PUnits, k=len(df['ProductionUnit']))
    new_PIds = random.choices(PIds, k=len(df['Prod_Id']))

    df['ProductionUnit'] = new_PUnits
    df['Prod_Id'] = new_PIds

    #TODO random start and end time

    df.to_json(outSchedule, indent=4)
    (loss, z) = officialScorer(outRoot, weeks[week]) #week is default #TODO can use breakdown to optimize
    # print(f'loss: {loss}: breakdown: {breakdown}')
    if loss > best:
        best = loss
        df.to_json(bestSchedule, indent=4)
        df.to_csv('tag_value_pairs.csv')
        patients = 0
    
    print('\\n\\n')
# enablePrsint() ########################################################

print(f'Best Score achived: {best}')
temp = pd.read_json(bestSchedule)
temp.to_json(outSchedule, indent=4)
officialScorer(outRoot, weeks[week])


