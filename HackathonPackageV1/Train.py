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


EPOCS = 1000
anger = 100

#Static paths
csv = 'tag_value_pairs.csv'
staticPath = f'HackathonPackageV1\\DataCache\\OptimizerSituations\\{weeks[week]}\\planningSchedule.json'
root = 'HackathonPackageV1\\DataCache\\OptimizerSituations'
InitialPaths = glob.glob(root + f'\\{weeks[week]}\\*.json')
temp = {}
for i in InitialPaths:
    temp[os.path.basename(i)] = i
InitialPaths = temp   # print(InitialPaths['initialPOs.json']) #How to get

#Output Paths
outRoot = 'HackathonPackageV1\\PredDataCache\\OptimizerSituations'
outSchedule = f'HackathonPackageV1\\PredDataCache\\OptimizerSituations\\{weeks[week]}\\planningSchedule.json'

#TempPaths
bestSchedule = 'HackathonPackageV1\\BestSchedule.json'

df = pd.read_json(staticPath)
IData = pd.read_json(InitialPaths['initialPOs.json'])

### SKU Dicts ###
SKUDict = pd.read_json(InitialPaths['SKU_Pull_Rate_Dict.json'])

### Catagorical Data ###
PUnits = IData['ProductionUnit'].unique()
PIds = IData['Prod_Id'].unique()


### Float Values ###
ForeStartMIN = IData['ForecastStartTime'].min()
ForeStartMAX = IData['ForecastStartTime'].max()

ForeEndMIN = IData['ForecastEndTime'].min()
ForeEndMAX = IData['ForecastEndTime'].max()
 

### Disable printing commands ###
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

#################################
allRuns = {}
# each ProductionUnit in planningSchedule.json will have a chance to draw 1 of the 13 ProductionUnit in InitialPOs.json
#this is a dict like this
"""
ProductionUnit 1: {prob1, prob2, ..., prob 13}
ProductionUnit 2: {prob1, prob2, ..., prob 13}
...
ProductionUnit 84: {prob1, prob2, ..., prob 13}
"""
machineProbWeights = {}

# similar for Prod_Id just refrencing SKU_Pull_Rate_Dict.json at key ProductionUnit n
"""
Prod_Id 1 given ProductionUnit 1: {prob1, prob2, ..., prob n} n == size of ProductionUnit 1 in InitialPOs.json
Prod_Id 2 given ProductionUnit 2: {prob1, prob2, ..., prob n} n == size of ProductionUnit 2 in InitialPOs.json
...
Prod_Id n given ProductionUnit n: {prob1, prob2, ..., prob n} n == size of ProductionUnit n in InitialPOs.json
"""
prod_IDWeights = {}






df.to_json(outSchedule, indent=4)
best = -364453 ## no change
iterations = 0
anxiety = 0
while iterations < EPOCS and anxiety < anger:

    # Stop Roll_inventory_Opimizer_Scoring from printing
    blockPrint() ##########################################################
    iterations += 1
    anxiety += 1
    

    ### randomizer ###
    # set list of ProducitonUnits in planningSchedule.json to a random producitonUnit from initialPOs.json
    df['ProductionUnit'] = random.choices(PUnits, k=len(df['ProductionUnit']))

    # given the productionUnit at key, set Prod_ID at key to random Prod_ID under given ProductionUnit in SKU_Pull_Rate_Dict.json
    for key, val in df['ProductionUnit'].items():
        df['Prod_Id'][key] = random.choice(SKUDict[val].dropna().keys())
        df.to_csv(csv)

    ######
    #TODO create optimizer to make educated guesses for 'good' production unit and Prod_ID combos

    # write to officialScorer input file
    df.to_json(outSchedule, indent=4)
    try:
        (loss, z) = officialScorer(outRoot, weeks[week]) #week is default #TODO can use breakdown to optimize
    except:
        continue

    # allRuns.update((loss, df))

    # Save best schedule
    # if loss > best:
    #     best = loss
    #     # save best schedule to seperate file
    #     df.to_json(bestSchedule, indent=4)
    #     patients = 0
    
    print('\n\n')
    enablePrint() ########################################################
    print(f'anxiety: {anxiety} :: iterations: {iterations}')

print(f'\n\n\nBest Score achived: {best}')


