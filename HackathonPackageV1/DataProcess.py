import json
import numpy as np
import pandas as pd
import glob

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
x = []

def iterate_nested_json_recursive(json_obj):
    for key, value in json_obj.items():
        if isinstance(value, dict):
            iterate_nested_json_recursive(value)
        else:
            x.append((key,value))

for d in date:
    for s in situation:
        extractPath = f'HackathonPackageV1/EX_DataCache/OptimizerSituations/{d}/{s}'
        file = (extractPath + '.json')
        with open(file, 'r') as f:
            iterate_nested_json_recursive(json.load(f))
# Print contents of x
# for key,val in x:
#     print(f'{key}------------{val}')