import json
import numpy as np
import pandas as pd

# Load the JSON data

date = ["2024-09-06 Week 1","2024-09-06 Week 2","2024-09-06 Week 3"]
situation = ["planningSchedule",
    "initialPOs",
    "reservedTimes",
    "plannedDemandConverting",
    "plannedDemandTM",
    "inventoryGradeCount",
    "planningRateDict",
    "SKU_Pull_Rate_Dict",
    "SKU_Converting_Specs_Dict",
    "SKU_TM_Specs",
    "scrapFactor",
    "currentTimeUTC"]

for d in date:
    # for n,s in enumerate(situation):
    for s in situation:
        with open("E:/Hackathon/GitProject/KCO/HackathonPackageV1/EX_DataCache/OptimizerSituations/"
                +f"{d}/{s}.json", 'r') as f:
            # globals()["var%d"%n] = json.load(f)
            # print(globals()["var" + str(n)])
            globals()["var_%s"%s] = json.load(f)
            # print(globals()["var_" + s])

        # data = json.load("E:\Hackathon\GitProject\KCO\HackathonPackageV1\EX_DataCache\OptimizerSituations\2024-09-06 Week 1\currentTimeUTC.json")
        # print(data)

# print(globals()["var_initialPOs"])

#Seperate data files into just numbers ["key": 5]
for s in situation:
    for key,val in globals()["var_%s"%s].items():
        print(key, val)
