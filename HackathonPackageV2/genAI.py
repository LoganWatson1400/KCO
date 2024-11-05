
# Team: BoylandGPT
# Members: Logan Watson, Jack Harmer
# AI Used: ChatGPT
#
# **NOTE**
# This project went through many iterations from a Neural Network to decision tree to genetic algorithm.
# This current model is heavily based off of recommendations/code created by ChatGPT.
# If you have any specific questions regarding this, then you can contact us at these emails...
# lawatson@uwm.edu OR pjharmer@uwm.edu

# ***Requirements.txt was altered, however, you may find that you wont need the added libraries
#@see README.md


import glob
import os
import pandas as pd
import random
import hashlib
import time
import gc
import json  # Import JSON module for saving/loading weights
from Roll_Inventory_Optimizer_Scoring import officialScorer
from datetime import timedelta

# Constants
week = 0
weeks = [
    '2024-09-06 Week 1',
    '2024-09-06 Week 2',
    '2024-09-06 Week 3'
]
score_cache = {}

MAX_TIME = 28000
POPULATION_SIZE = 10
TOURNAMENT_SIZE = 5
ELITE_COUNT = 3
BASE_MUTATION_RATE = 0.2
MIN_MUTATION_RATE = 0.05
MAX_MUTATION_RATE = 0.5
TIME_SHIFT_MIN = 1 #    HOURS
TIME_SHIFT_MAX = 10 #3 # HOURS

MIN_DATE = 1725667200000
MIN_DATE = pd.to_datetime(MIN_DATE, unit='ms')
MAX_DATE = 1726531199000
MAX_DATE = pd.to_datetime(MAX_DATE, unit='ms')


# Paths and initial data load
root = 'HackathonPackageV2\\DataCache\\OptimizerSituations'
staticPath = f'{root}\\{weeks[week]}\\planningSchedule.json'
InitialPaths = {os.path.basename(path): path for path in glob.glob(root + f'\\{weeks[week]}\\*.json')}
outRoot = 'HackathonPackageV2\\PredDataCache\\OptimizerSituations'
outSchedule = f'HackathonPackageV2\\PredDataCache\\OptimizerSituations\\{weeks[week]}\\planningSchedule.json'
weights_filename = 'weights.json'  # Filename to save/load weights

# Load initial data
df = pd.read_json(staticPath)
IData = pd.read_json(InitialPaths['initialPOs.json'])
SKUDict = pd.read_json(InitialPaths['SKU_Pull_Rate_Dict.json'])
reservedTimes = pd.read_json(InitialPaths['reservedTimes.json'])
df['Prod_Id']
df['ForecastStartTime']
df['ForecastEndTime']
df['ForecastQuantity']
df['ProductionUnit']


# Load weights from file if it exists
def load_weights():
    if os.path.exists(weights_filename):
        with open(weights_filename, 'r') as file:
            return json.load(file)
    return [1.0] * POPULATION_SIZE  # Default weights if no file exists

# Save weights to a file
def save_weights(weights):
    with open(weights_filename, 'w') as file:
        json.dump(weights, file)

def isIn(x, arr):
    for y in arr:
        if y == x:
            return True
    return False

def getUsed(PU):
    sch = pd.read_json(outSchedule)
    reseved = pd.read_json(InitialPaths['reservedTimes.json'])
    dfPU = {}
    for key, val in sch['ProductionUnit'].items():
        tmp = []  
        for key, val2 in sch['ProductionUnit'].items():
            if val != val2:
                continue
            tmp.append(key)  
        dfPU.update({val : tmp})

    used = []
    
    for key, val in sch['ForecastStartTime'].items():
        reserved = isIn(key, reseved)
        if reserved or isIn(key, dfPU[PU]):
            if reserved:
                used.append({'start' : sch['ForecastStartTime'][key], 'end' : sch['ForecastEndTime'][key]})    
            else:
                used.append({'start' : val, 'end' : sch['ForecastEndTime'][key]})
    return used

def dateBetween(start1, start2, end2, end1):
    return start1 <= start2 and end2 < end1 # can equill the end, just not before it

def hasOverlap(dfStart, dfEnd, PU):
    used = getUsed(PU)
    for u in used:
        if dateBetween(pd.to_datetime(u['start'], unit='ms'), dfStart, dfEnd, pd.to_datetime(u['end'], unit='ms')):
            return True
    return False

def removeAllOverlap(schedule):
    if not isinstance(schedule, pd.DataFrame):
        print('your a fool')
        return None
    for key, val in schedule['ProductionUnit'].items():
        unit = val
        start = pd.to_datetime(schedule['ForecastStartTime'][key], unit='ms')
        end = pd.to_datetime(schedule['ForecastEndTime'][key], unit='ms')
        while hasOverlap(start, end, unit):
            start = pd.to_datetime(MIN_DATE.value, unit='ms')
            end = start + pd.Timedelta(hours=2) # TODO at least 2
        schedule['ForecastStartTime'][key] = int((start + pd.Timedelta(hours=2)).value / 1e6)
        schedule['ForecastEndTime'][key] = int((end + pd.Timedelta(hours=2)).value / 1e6)
    return schedule

def readJson():
    return pd.read_json(staticPath)

def updateJson(var):
    var.to_json(outSchedule, indent=4)

def newScore(data):
    hash_val = hash_schedule(data)
    if hash_val in score_cache:
        return score_cache[hash_val]
    
    updateJson(data)
    score = officialScorer(outRoot, weeks[week])
    score_cache[hash_val] = score
    return score

def hash_schedule(schedule):
    return hashlib.md5(pd.util.hash_pandas_object(schedule).values).hexdigest()

def mutate(schedule, mutation_rate, weights):
    mutation_strength = 2 if random.random() < mutation_rate else 1
    for _ in range(mutation_strength):
        row = random.choice(schedule.index)
        production_unit = schedule.at[row, 'ProductionUnit']
        
        # Mutate Prod_Id
        if production_unit in SKUDict.columns:
            possible_prod_ids = SKUDict[production_unit].dropna().keys().tolist()
            if possible_prod_ids:
                new_prod_id = random.choice(possible_prod_ids)
                schedule.at[row, 'Prod_Id'] = new_prod_id

        start_time = schedule.at[row, 'ForecastStartTime']
        start_time = pd.to_datetime(start_time, unit='ms')
        time_shift = random.randint(TIME_SHIFT_MIN, TIME_SHIFT_MAX)  # Hours after ForecastEndTime 
        new_start_time = start_time + pd.Timedelta(hours=time_shift)

        # Check against reserved times
        while new_start_time + pd.Timedelta(hours=2) >=  MAX_DATE or hasOverlap(new_start_time, new_start_time + pd.Timedelta(hours=2), production_unit):
            if new_start_time + pd.Timedelta(hours=2) >=  MAX_DATE:
                new_start_time = MIN_DATE
            else:
                new_start_time += pd.Timedelta(hours=1)


        schedule.at[row, 'ForecastStartTime'] = int(new_start_time.value / 1e6)
        schedule.at[row, 'ForecastEndTime'] = int((new_start_time + pd.Timedelta(hours=2)).value / 1e6)


        # Mutate ForecastQuantity
        quantity = schedule.at[row, 'ForecastQuantity']
        if pd.notna(quantity):
            adjustment_factor = random.uniform(-0.1, 0.1)
            new_quantity = quantity * (1 + adjustment_factor)
            schedule.at[row, 'ForecastQuantity'] = max(0, round(new_quantity))

    return schedule

def crossover(parent1, parent2):
    child = parent1.copy()
    for i in range(len(child)):
        if random.random() < 0.5:
            child.iloc[i] = parent2.iloc[i]
    return child

def tournament_selection(population):
    tournament = random.sample(population, TOURNAMENT_SIZE)
    return min(tournament, key=lambda x: x['score'])['schedule']

def save_best_schedule(best_schedule, filename):
    best_schedule.to_json(filename, indent=4)

def calculate_mutation_weights(population):
    # This function expects a list of dictionaries with a 'score' key in each dictionary
    return [1.0 / abs(ind['score']) if ind['score'] != 0 else 1.0 for ind in population]

def update_weights(best_score):
    # Update weights based on best score
    weights = load_weights()
    updated_weights = [weight * (1 - best_score / abs(best_score)) for weight in weights]
    save_weights(updated_weights)

def genetic_algorithm():
    total_time = 0
    initial_schedule = readJson().copy()  # Read initial schedule
    initial_schedule = removeAllOverlap(initial_schedule)
    population = [{'schedule': mutate(initial_schedule.copy(), BASE_MUTATION_RATE, [1.0] * len(initial_schedule)), 'score': None} for _ in range(POPULATION_SIZE)]

    generation = 0
    previous_best_score = float('inf')

    while total_time < MAX_TIME:
        generation += 1

        # Check for overlapping intervals
        # if has_overlapping_intervals(df, reservedTimes):
        #     pass

        # Evaluate scores with a timeout
        for individual in population:
            if individual['score'] is None:
                start_time = time.time()
                try:
                    individual['score'], _ = newScore(individual['schedule'])
                except TimeoutError:
                    print(f"Timeout: Skipping schedule evaluation for this individual in Generation {generation}.")
                    individual['score'] = float('inf')
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                print(f"Generation {generation}, Evaluated Score: {individual['score']} (Time taken: {elapsed_time:.2f}s)")
                if total_time >= MAX_TIME:
                    break

        # Handle populations with valid scores
        valid_population = [ind for ind in population if ind['score'] is not None]

        if not valid_population:
            print("No valid scores found in the population.")
            break

        # Sort and apply elitism
        valid_population.sort(key=lambda x: abs(x['score']))
        best_score = valid_population[0]['score']
        print(f"Generation {generation}, Best Score: {best_score}")

        # Adjust mutation rate dynamically
        mutation_rate = max(MIN_MUTATION_RATE, BASE_MUTATION_RATE * (1 - best_score / previous_best_score)) if best_score < previous_best_score else min(MAX_MUTATION_RATE, BASE_MUTATION_RATE * (1 + best_score / previous_best_score))
        previous_best_score = best_score

        # Calculate mutation weights based on the current population scores
        weights = calculate_mutation_weights(valid_population)

        # Apply elitism and generate new population
        elite_individuals = valid_population[:ELITE_COUNT]
        new_population = elite_individuals.copy()

        # Generate offspring
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(valid_population[:POPULATION_SIZE // 2], 2)
            child_schedule = crossover(parent1['schedule'], parent2['schedule'])
            new_population.append({'schedule': mutate(child_schedule, mutation_rate, weights), 'score': None})

        # Save weights based on performance
        update_weights(best_score)

        # Update the population for the next generation
        population = new_population

    best_schedule = valid_population[0]['schedule']
    save_best_schedule(best_schedule, outSchedule)

# genetic_algorithm()
# score,breakdown = officialScorer(outRoot, weeks[week])
# print(score)
# print(breakdown)

######################
# for i in range(3):

#     week = i
#     genetic_algorithm()
#     score,breakdown = officialScorer(outRoot, weeks[week])
#     print(score)

#     print('=============================================')
#     for key, _ in breakdown.items():
#         print(f'{key}')
#     print('\n')
#     for _, val in breakdown.items():
#         print(f'{val}')
#     print('=============================================')
#     i += 1

genetic_algorithm()
score,breakdown = officialScorer(outRoot, weeks[week])
print(score)

print('=============================================')
for key, _ in breakdown.items():
    print(f'{key}')
for _, val in breakdown.items():
    print(f'{val}')