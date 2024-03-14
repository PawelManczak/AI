from itertools import compress
import random
import time
import matplotlib.pyplot as plt
import numpy as np

from data import *


def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]


def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))


def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


def get_sum_of_fitness():
    msum = 0
    for individual in population:
        msum += fitness(items, knapsack_max_capacity, individual)
    return msum


def get_probability(individual):
    individualProbability = fitness(items, knapsack_max_capacity, individual)
    return individualProbability / get_sum_of_fitness()


def get_parents():
    # Lista z rozkładem prawdopodobieństwa dla każdego osobnika
    probabilities = list()
    for individual in population:
        probabilities.append(get_probability(individual))
    parents = list()

    # Funkcja losująca osobnika z uwzględnieniem obliczonego prawdopodobieństwa
    def random_individual():
        rand = random.randint(0, 99999999) / 100000000
        weight_sum = 0.0
        for idx, probability in enumerate(probabilities):
            weight_sum += probability
            if (weight_sum > rand): return population[idx]
        # should never reach this point
        return population[0]

    for _ in range(n_selection):
        parents.append(random_individual())

    return parents


def create_children(parents: list):
    def get_children(parent_a: list, parent_b: list):
        chunks_A, chunks_B = np.split(np.array(parent_a), 2), np.split(np.array(parent_b), 2)
        return np.append(chunks_A[0], chunks_B[1]).tolist(), np.append(chunks_A[1], chunks_B[0]).tolist()

    # Lista dzieci pokolenia
    children: list = []
    # 3-wymiarowa macierz automatycznie dopasowująca rodziców w pary
    fancy_array = np.array(parents).reshape(-1, 2, len(parents[0]))
    for pair in fancy_array:
        child_A, child_B = get_children(pair[0], pair[1])
        children.append(child_A)
        children.append(child_B)
    return children


def mutate_children(children: list):
    for child in children:
        rand_index = random.randint(0, len(children) - 1)
        child[rand_index] = not child[rand_index]


def get_best():
    better_array = np.array([fitness(items, knapsack_max_capacity, specimen) for specimen in population])
    res = []
    for _ in range(n_elite):
        best_index = better_array.argmax()
        res.append(population[best_index][:])  # Przekazanie listy przez wartość
        # Przepraszam wszystkich purystów za to co się teraz stanie, nie chce mi się juz robic tego ladnie xd
        better_array = better_array.tolist()
        better_array.remove(better_array[best_index])
        better_array = np.array(better_array)
    return res


items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 200
n_selection = 20
n_elite = 1

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []

# 1
population = initial_population(len(items), population_size)

for _ in range(generations):
    population_history.append(population)

    # TODO: implement genetic algorithm
    best_specimen = get_best()
    parents = get_parents()
    children = create_children(parents)
    mutate_children(children)
    population = children
    for elite in best_specimen:
        population.append(elite)

    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])

plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
