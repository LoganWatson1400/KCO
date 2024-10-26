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

# one-hot encoding
boolVals = [
    ('Complete',    np.array([1,0])),
    ('Incomplete',  np.array([0,1])),
   ( 'Active', bool(1))
    ]
grades = [
    ('Grade1', np.array([1,0,0,0,0,0])),
    ('Grade2', np.array([0,1,0,0,0,0])),
    ('Grade3', np.array([0,0,1,0,0,0])),
    ('Grade4', np.array([0,0,0,1,0,0])),
    ('Grade5', np.array([0,0,0,0,1,0])),
    ('Grade6', np.array([0,0,0,0,0,1]))
]

ProductionUnit = [
        ('BI4 Machine',         np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])),
        ('BI4 Machine',         np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])),
        ('CFR1 Parent Rolls',   np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])),
        ('L07 Winder',          np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])),
        ('L08 Winder',          np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])),
        ('L09 Winder',          np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])),
        ('L10 Winder',          np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])),
        ('L11 Winder',          np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])),
        ('PB1 Winder',          np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])),
        ('PB2 Winder',          np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])),
        ('PB3 Winder',          np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])),
        ('PB4 Winder',          np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])),
        ('PB5 Winder',          np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])),
        ('PB6 Winder',          np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])),
        ('TM3 Machine',         np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]))
]

strVals = [
    boolVals,
    grades,
    ProductionUnit
]

x = []
notInt = []

def findInStrVal(val):
    for arr in strVals:
        for i,v in arr:
            if val == i:
                return v
    return '-1'

def iterate_nested_json_recursive(json_obj):
    for key, value in json_obj.items():
        try: 
            value = float(value)
        except:
            pass
        
        #search dictionaries recursivly
        if isinstance(value, dict):
            iterate_nested_json_recursive(value)
        elif not isinstance(value, str):
            x.append((key,value)) 
        else:
            # look for value in hard coded string conversion
            found = findInStrVal(value)
            if found != '-1':
                x.append((key,found))


for d in date:
    for s in situation:
        extractPath = f'HackathonPackageV1/EX_DataCache/OptimizerSituations/{d}/{s}'
        file = (extractPath + '.json')
        with open(file, 'r') as f:
            iterate_nested_json_recursive(json.load(f))
# Print contents of x
Xval = []
for key,val in x:
    print(f'{key}------------{val}')
    Xval.append(val)


