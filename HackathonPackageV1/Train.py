from copy import deepcopy
import glob
import os
import sys
import pandas as pd
import numpy as np
from numpy.random import choice
import random
from Roll_Inventory_Optimizer_Scoring import officialScorer

week = 0
weeks = [
    '2024-09-06 Week 1',
    '2024-09-06 Week 2',
    '2024-09-06 Week 3'
]


EPOCS = 1000000
anger = 1000000
Batch = 1

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
allRuns = []
# each ProductionUnit in planningSchedule.json will have a chance to draw 1 of the 13 ProductionUnit in InitialPOs.json
#this is a dict like this
"""
prod unit 1 (run 100, unit 1 occurances / runs   then scale by loss)

ProductionUnit 1: {prob1, prob2, ..., prob 13} 
ProductionUnit 2: {prob1, prob2, ..., prob 13}
...
ProductionUnit 84: {prob1, prob2, ..., prob 13}
"""
temp = {}
for key, val in IData.get('ProductionUnit', {}).items():
    temp.update({val : {'occurances': 1, 'SUM loss': -14000000}})
mpw = {unit: deepcopy(temp) for unit in df['ProductionUnit'].keys()}

temp = {}
for key, val in IData.get('ProductionUnit', {}).items():
    temp.update({val : 0})
probWeights = {unit: deepcopy(temp) for unit in df['ProductionUnit'].keys()}


# for key,val in df['ProductionUnit'].items():
#     mpw.update({key : UnitW})
    # mpw.setdefault(key, UnitW.copy())
    # mpw.setdefault(key, UnitW)
# pd.DataFrame(mpw).to_json('HackathonPackageV1\mpw.json', indent=4)

pd.DataFrame(mpw).to_json('HackathonPackageV1\mpw.json', indent=4)
# print('test')

# similar for Prod_Id just refrencing SKU_Pull_Rate_Dict.json at key ProductionUnit n
"""
Prod_Id 1 given ProductionUnit 1: {prob1, prob2, ..., prob n} n == size of ProductionUnit 1 in InitialPOs.json
Prod_Id 2 given ProductionUnit 2: {prob1, prob2, ..., prob n} n == size of ProductionUnit 2 in InitialPOs.json
...
Prod_Id n given ProductionUnit n: {prob1, prob2, ..., prob n} n == size of ProductionUnit n in InitialPOs.json
"""





df.to_json(outSchedule, indent=4)
best = -364453 ## no change
iterations = 0
anxiety = 0
probs = {}
while iterations < EPOCS and anxiety < anger:

    # Stop Roll_inventory_Opimizer_Scoring from printing
    blockPrint() ##########################################################

    ### Educated Guesses ###
    probWeights = {}
    if iterations%Batch == 0 or iterations == 1:

        occ = 0
        avgLoss = 0
        for k, v in df['ProductionUnit'].items():
            totalWeight = 0
            probWeights.update({k : {}})
            for key, val in mpw[k].items():
                sumLoss = mpw[k][key]['SUM loss']
                occ = mpw[k][key]['occurances']
                if occ != 0:
                    avgLoss = sumLoss / occ * -1
                else:
                    avgLoss = 0

                totalWeight += (avgLoss * occ)

            for key, val in mpw[k].items():
                sumLoss = mpw[k][key]['SUM loss']
                occ = mpw[k][key]['occurances']
                if occ != 0:
                    avgLoss = sumLoss / occ * -1
                else:
                    avgLoss = 0

                prob = [(avgLoss * occ) / totalWeight]
                probWeights[k].update({key : prob})
        pd.DataFrame(probWeights).to_json('HackathonPackageV1\probWeights.json', indent=4)
        probs = {}
        # PUnits = []
        hasFound = False
        for key, val in probWeights.items():
            temparr = []
            for x, y in probWeights[key].items():
                temparr.append(y[0])
            probs.update({key : temparr})

    # set list of ProducitonUnits in planningSchedule.json to a random producitonUnit from initialPOs.json
    test = ['BI4 Machine', 'CFR1 Parent Rolls', 
            'L07 Winder', 'L08 Winder', 'L09 Winder', 
            'L10 Winder', 'L11 Winder', 'PB1 Winder', 
            'PB2 Winder', 'PB3 Winder', 'PB4 Winder', 
            'PB5 Winder', 'PB6 Winder', 'TM3 Machine'
            ]
    
    if iterations != 0:
        for key, val in df['ProductionUnit'].items():
            df['ProductionUnit'][key] = choice(test, p=probs[key])


        # given the productionUnit at key, set Prod_ID at key to random Prod_ID under given ProductionUnit in SKU_Pull_Rate_Dict.json
        for key, val in df['ProductionUnit'].items():
            df['Prod_Id'][key] = random.choice(SKUDict[val].dropna().keys())
            # df.to_csv(csv)

    # write to officialScorer input file
    df.to_json(outSchedule, indent=4)
    try:
        (loss, z) = officialScorer(outRoot, weeks[week]) #week is default #TODO can use breakdown to optimize
    except:
        continue   

    for key, val in df['ProductionUnit'].items():
        mpw[key][val]['occurances'] += 1
        mpw[key][val]['SUM loss'] += loss

    # Save best schedule
    if loss > best or iterations == 1:
        best = loss
        # save best schedule to seperate file
        pd.DataFrame(probWeights).to_json('HackathonPackageV1\BestprobWeights.json', indent=4)
        df.to_json(bestSchedule, indent=4)
        patients = 0
        anxiety = 0
    else:
        anxiety += 1

    iterations += 1


    mpw.to_json('HackathonPackageV1\mpw.json', indent=4)
    print('\n\n')
    enablePrint() ########################################################
    print(f'anxiety: {anxiety} :: iterations: {iterations} :: Best: {best} :: Current: {loss}')

print(f'\n\n\nBest Score achived: {best}')


