
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
week = 2
weeks = [
    '2024-09-06 Week 1',
    '2024-09-06 Week 2',
    '2024-09-06 Week 3'
]
MAX_TIME = 600
POPULATION_SIZE = 10
GENERATIONS = 50
MUTATION_RATE = 0.7
TOURNAMENT_SIZE = 5
ELITE_COUNT = 3
BASE_MUTATION_RATE = 0.2
MIN_MUTATION_RATE = 0.05
MAX_MUTATION_RATE = 0.5

# Paths and initial data load
root = 'HackathonPackageV1\\DataCache\\OptimizerSituations'
staticPath = f'{root}\\{weeks[week]}\\planningSchedule.json'
InitialPaths = {os.path.basename(path): path for path in glob.glob(root + f'\\{weeks[week]}\\*.json')}
outRoot = 'HackathonPackageV1\\PredDataCache\\OptimizerSituations'
outSchedule = f'HackathonPackageV1\\PredDataCache\\OptimizerSituations\\{weeks[week]}\\planningSchedule.json'
weights_filename = 'weights.json'  # Filename to save/load weights

# Load initial data
df = pd.read_json(staticPath)
IData = pd.read_json(InitialPaths['initialPOs.json'])
SKUDict = pd.read_json(InitialPaths['SKU_Pull_Rate_Dict.json'])
reservedTimes = pd.read_json(InitialPaths['reservedTimes.json'])

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

# Utility function to detect interval overlap
def has_overlapping_intervals(schedule, reserved_times):
    for _, row in schedule.iterrows():
        start_time = row['ForecastStartTime']
        end_time = row['ForecastEndTime']
        overlaps = reserved_times.apply(
            lambda r: r['ForecastStartTime'] < end_time and start_time < r['ForecastEndTime'],
            axis=1
        ).any()
        if overlaps:
            return True
    return False

# Attempt to convert values in SKUDict to float32
def convert_to_float(x):
    if isinstance(x, dict):
        return float(x.get('some_key', 0))  # Replace 'some_key' with the appropriate key
    return pd.to_numeric(x, errors='coerce')

SKUDict = SKUDict.applymap(convert_to_float).astype('float32')

# Caching scored schedules
score_cache = {}

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
        
        # Mutate ForecastStartTime and ForecastEndTime based on initialPOs
        try:
            start_time = pd.to_datetime(IData['ForecastEndTime'][production_unit], errors='coerce')
            if pd.notna(start_time):
                # Pick a new start time after the initial POs end and not in reserved times
                time_shift = random.randint(1, 3)  # Hours after ForecastEndTime
                new_start_time = start_time + pd.Timedelta(hours=time_shift)

                # Check against reserved times
                while has_overlapping_intervals(pd.DataFrame([{'ForecastStartTime': new_start_time, 'ForecastEndTime': new_start_time + pd.Timedelta(hours=1)}]), reservedTimes):
                    new_start_time += pd.Timedelta(hours=1)

                schedule.at[row, 'ForecastStartTime'] = new_start_time
                schedule.at[row, 'ForecastEndTime'] = new_start_time + pd.Timedelta(hours=1)
        except KeyError:
            continue  # Skip if production unit not found in initialPOs

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
    population = [{'schedule': mutate(initial_schedule.copy(), BASE_MUTATION_RATE, [1.0] * len(initial_schedule)), 'score': None} for _ in range(POPULATION_SIZE)]
    
    generation = 0
    previous_best_score = float('inf')

    while total_time < MAX_TIME:
        generation += 1

        # Check for overlapping intervals
        if has_overlapping_intervals(df, reservedTimes):
            pass

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

# Run genetic algorithm
genetic_algorithm()
def genetic_algorithm_fixed_generations(max_generations=50):
    total_time = 0
    initial_schedule = readJson().copy()  # Read initial schedule
    population = [{'schedule': mutate(initial_schedule.copy(), BASE_MUTATION_RATE, [1.0] * len(initial_schedule)), 'score': None} for _ in range(POPULATION_SIZE)]
    
    generation = 0
    previous_best_score = float('inf')

    while generation < max_generations:
        generation += 1

        # Check for overlapping intervals
        if has_overlapping_intervals(df, reservedTimes):
            pass

        # Evaluate scores without timeout
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

# Run genetic algorithm for a fixed number of generations
# genetic_algorithm_fixed_generations(max_generations=50)  # You can set the desired number of generations here
score,breakdown = officialScorer(outRoot, weeks[week])
print(score)
print(breakdown)
