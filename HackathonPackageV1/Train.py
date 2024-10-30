import glob
import os
import sys
import pandas as pd
import numpy as np
import random
from Roll_Inventory_Optimizer_Scoring import officialScorer
import queue as q

week = 0
weeks = [
    '2024-09-06 Week 1',
    '2024-09-06 Week 2',
    '2024-09-06 Week 3'
]

MAX_TIME = 600 #10min
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


# best = -364453 ## no change
# iterations = 0
# anxiety = 0
# while iterations < EPOCS and anxiety < anger:

#     # Stop Roll_inventory_Opimizer_Scoring from printing
#     blockPrint() ##########################################################
#     iterations += 1
#     anxiety += 1
    

#     ### randomizer ###
#     # set list of ProducitonUnits in planningSchedule.json to a random producitonUnit from initialPOs.json
#     df['ProductionUnit'] = random.choices(PUnits, k=len(df['ProductionUnit']))

#     # given the productionUnit at key, set Prod_ID at key to random Prod_ID under given ProductionUnit in SKU_Pull_Rate_Dict.json
#     for key, val in df['ProductionUnit'].items():
#         df['Prod_Id'][key] = random.choice(SKUDict[val].dropna().keys())
#         df.to_csv(csv)

#     ######
#     #TODO create optimizer to make educated guesses for 'good' production unit and Prod_ID combos



#     # write to officialScorer input file
#     df.to_json(outSchedule, indent=4)
#     (loss, z) = officialScorer(outRoot, weeks[week]) #week is default #TODO can use breakdown to optimize

#     # Save best schedule
#     if loss > best:
#         best = loss
#         # save best schedule to seperate file
#         df.to_json(bestSchedule, indent=4)
#         patients = 0
    
#     print('\n\n')
#     enablePrint() ########################################################
#     print(f'anxiety: {anxiety} :: iterations: {iterations}')
# print(f'\n\n\nBest Score achived: {best}')

# (best, z) = officialScorer(outRoot, weeks[week])
# print(f'\n\n\nBest Score achived: {best}')

# EX: Breakdown
# HoursBelowTMPOMinimumRuntime2hr                -0.000000
# totalOverlapHours                         -137211.111111
# gradeOOPViolationHoursTotal                -41000.000000
# gradeChangesConvertingHours                -40627.777778
# hoursBeyondEndBoundary                         -0.000000
# gradeMinViolationHoursTotal                -46000.000000
# proposedDemandViolationPenaltyTotal       -293473.056385
# HoursBelowTMPOMinimumRuntime8hr              -900.805556
# HoursBelowConvertingPOMinimumRuntime8hr     -2997.083333
# TM4GradeChangeOrderViolations                   0.000000
# gradeMaxViolationHoursTotal                 -7400.000000
# GradeChangeCountTM                            -30.000000
# GradeChangeCountConverting                    -55.000000
# rollsBelowMaxInventoryScoreTotal              -21.104258


def compareOfficialScores(a,b):
    if a > b:
        return 1
    elif a < b:
        return -1
    else:
        return 0

###JSON###
def readJson():
    var = pd.read_json(staticPath)
    return var

def updateJson(var):
    var.to_json(outSchedule, indent=4)

###Get new score###
def newScore(data):
    updateJson(data)
    (loss,breakdown) = officialScorer(outRoot, weeks[week])
    return loss,breakdown

####Console print####
def toConsole(*arg):
    # pass
    print(f"Epoch: {arg[0]} --- Current loss:{arg[1]} --- Improvement percent:{arg[2]}")

# Pick values which are likely to change the loss
#Replace placeholder vals with pandas dataframe of stuff
# jsonData = readJson()
# machine_df = 5 #Placeholder
# prodID_df = 5 #Placeholder

# r1,c1 = machine_df.shape
# r2,c2 = prodID_df.shape

# #Populate weights with random values in range (-0.3, 0.3)
# machineProbWeights = np.random.uniform(-0.3, 0.3, (r1,c1)).tolist()
# prod_IDWeights = np.random.uniform(-0.3, 0.3, (r2,c2)).tolist()
# validHeaders = ["ForecastStartTime","ForecastEndTime","ForecastQuantity"]

import timeit as t
import datetime as d
from array import array

###Main Algorithm###
def algorithm():

    jsonData = readJson()

    rows,cols = jsonData.shape

    #Populate weights with random values in range (-0.3, 0.3)
    machineProbWeights = np.random.uniform(-0.3, 0.3, (rows,1)).tolist()
    prod_IDWeights = np.random.uniform(-0.3, 0.3, (rows,1)).tolist()

    #general vars
    header = {"ProductionUnit", "Prod_Id", "ForecastStartTime", "ForecastEndTime", "ForecastQuantity"}
    pastScores = array('d')
    curScore, curBreak = officialScorer(outRoot, weeks[week])
    duration_obj = d.datetime.strptime("00:30:00", '%H:%M:%S')

    #monitoring vars
    ##Loop
    start_time = t.default_timer()
    improving = True
    ##toConsole
    startScore = curScore
    epoch = 1

    def isImproving():
        x1 = 1
        x2 = len(pastScores)
        y1 = pastScores[0]
        y2 = pastScores[x2-1]
        m = (y2-y1)/(x2-x1)
        if(m > 0):
            return True
        else:
            return False

    def chooseIndex():
        random_row = random.choice(jsonData.index)
        random_col = random.choice(jsonData.columns)
        return (random_row,random_col)
    
    ## Main alg loop
    while (((t.default_timer() - start_time) < (MAX_TIME-30)) and improving):
        # set list of ProducitonUnits in planningSchedule.json to a random producitonUnit from initialPOs.json
        df['ProductionUnit'] = random.choices(PUnits, k=len(df['ProductionUnit']))

        # given the productionUnit at key, set Prod_ID at key to random Prod_ID under given ProductionUnit in SKU_Pull_Rate_Dict.json
        for key, val in df['ProductionUnit'].items():
            df['Prod_Id'][key] = random.choice(SKUDict[val].dropna().keys())
            # df.to_csv(csv)
        
        
        # Make new vals off of current losses (predict)
        # Get choices
        choice = chooseIndex()
        # Make high vals
        hiData = jsonData
        if isinstance(hiData.at[choice[0],choice[1]], d.datetime):
            hiData.at[choice[0],choice[1]] = hiData.at[choice[0],choice[1]] + d.timedelta(minutes=duration_obj.minute)
        elif isinstance(hiData.at[choice[0],choice[1]], float):
            hiData.at[choice[0],choice[1]] = hiData.at[choice[0],choice[1]] + 25

        # Make lo vals
        loData = jsonData
        if isinstance(hiData.at[choice[0],choice[1]], d.datetime):
            loData.at[choice[0],choice[1]] = loData.at[choice[0],choice[1]] - d.timedelta(minutes=duration_obj.minute)
        elif isinstance(hiData.at[choice[0],choice[1]], float):
            loData.at[choice[0],choice[1]] = loData.at[choice[0],choice[1]] - 25

        # official score new vals
        hiScore, hiBreak = newScore(hiData)

        #Compare new Scores high/low to current
        if(compareOfficialScores(hiScore, curScore) > 0):
            curScore = hiScore
            curBreak = hiBreak

        loScore, loBreak = newScore(loData)
        if(compareOfficialScores(loScore, curScore) > 0):
            curScore = loScore
            curBreak = loBreak

        # Update Past Score Queue
        if(len(pastScores) == 10):
            pastScores.pop(0)
            pastScores.append(curScore)
        else:
            pastScores.append(curScore)

        # CONDITIONAL check if improving (Timer/progress)
        if(len(pastScores) == 10):
            improving = isImproving()

        # Print Info
        toConsole(epoch,curScore,(curScore/startScore)*100)
        epoch+=1
    
    
# test = array('d')
# test.append(1)
# test.append(2)
# test.append(3)
# test.append(4)
# print(test.pop(0))
# print(test.pop(0))
algorithm()
print("Finished!")
# toConsole(1,2,3)