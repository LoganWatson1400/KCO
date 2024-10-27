import json
import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import OneHotEncoder

#to get data size do data.collums

# Load the JSON data
date = ['2024-09-06 Week 1','2024-09-06 Week 2','2024-09-06 Week 3']
situation = ['planningSchedule',
    'initialPOs',
    'reservedTimes',
    'plannedDemandConverting',
    'plannedDemandTM',
    'inventoryGradeCount',
    'planningRateDict',
    'SKU_Pull_Rate_Dict',
    'SKU_Converting_Specs_Dict',
    'SKU_TM_Specs',
    'scrapFactor',
    'currentTimeUTC']

# extractPath = f'HackathonPackageV1/DataCache/OptimizerSituations/{d}/{s}.json'
# exportPath = f'HackathonPackageV1/EX_DataCache/OptimizerSituations/{d}/{s}.json'


def __getData(d):
    data = {}
    for s in situation:
        extractPath = f'HackathonPackageV1/EX_DataCache/OptimizerSituations/{d}/{s}.json'
        file = __getDataSingle(extractPath)
        data.update(file)
    return data

def __getDataSingle(extractPath):
    data = {}
    for file in glob.glob(extractPath):
        with open(file) as f:
            data.update(json.load(f))
    return data

def __normalize(data):
    return pd.json_normalize(data)

def __clean_data(df):

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Encode numeric columns
    encoded_numeric = pd.get_dummies(df[numeric_cols]).reset_index(drop=True)
  
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Encode categorical data
    encoder = OneHotEncoder(sparse=False)
    encoded_categorical = encoder.fit_transform(df[categorical_cols])

    # Create encoded columns
    encoded_categorical_df = pd.DataFrame(encoded_categorical, 
                                       columns=[f'{col}_{val}' for col in categorical_cols for val in encoder.categories_[list(categorical_cols).index(col)]])
    encoded_categorical_df.index = df.index
    
    # Combine encoded data
    final_df = pd.concat([encoded_numeric, encoded_categorical_df], axis=1)

    return final_df

def toCV(data):
    data.to_csv('tag_value_pairs.csv', index=False)
    print('CSV file saved successfully.')

def processData(d):
    data = __getData(d)
    df = __normalize(data)
    # cleaned_df = df ## TODO bypasses cleaning function for convinience
    cleaned_df = __clean_data(df)
    return cleaned_df

def updatejson(d, preds):
    predNumber = 0
    for s in situation:
        extractPath = f'HackathonPackageV1/DataCache/OptimizerSituations/{d}/{s}.json'
        exportPath = f'HackathonPackageV1/EX_DataCache/OptimizerSituations/{d}/{s}.json'
        prevData = __getDataSingle(extractPath)
        
        #iterator for preds
        pVals = (item[0] for item in preds)

        
        for key in prevData:
            try:
                temp = next(pVals)
                prevData[key] = next(pVals)
            except StopIteration:
                print('',end='')

        with open(exportPath,'w') as json_file:
            json.dump(prevData, json_file, indent=4, separators= (',', ':'))


#working on this
# data = {}
# for s in situation:
#     extractPath = f'HackathonPackageV1/EX_DataCache/OptimizerSituations/{d}/{s}.json'
#     for file in glob.glob(extractPath):
#         with open(file) as f:
#             temp = json.load(f)
#             print(temp)
#             data.update(json.load(f))
#     data.update(file)

