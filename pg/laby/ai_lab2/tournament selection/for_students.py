from itertools import compress
import random
import time
import matplotlib.pyplot as plt
import numpy as np

from data import *

import numpy.random as npr


def get_best():
    population_fitness = np.array([fitness(items, knapsack_max_capacity, individual) for individual in population])
    elite = []
    for _ in range(n_elite):
        best_index = population_fitness.argmax()
        elite.append(population[best_index])
        population_fitness = population_fitness.tolist()
        population_fitness.remove(population_fitness[best_index])
        population_fitness = np.array(population_fitness)
    return elite


def mutate_children(children: list):
    for child in children:
        for i in range(0,26):
            mrand = random.randint(0,26)
            if mrand == 1:
                child[i] = not child[i]




def create_children(parents):
    m_parents = parents
    already_selected = 0
    children = []

    def get_children(parent_a: list, parent_b: list):
        randN = random.randint(0, 25)

        first_halfA = parent_a[:randN]
        sec_halfA = parent_a[randN:]
        first_halfB = parent_b[:randN]
        sec_halfB = parent_b[randN:]

        return np.append(first_halfA, sec_halfB).tolist(), np.append(first_halfB, sec_halfA).tolist()

    for _ in range(int(n_selection / 2) - 1):
        rand_index = random.randint(0, n_selection - already_selected - 1)
        parent1 = m_parents[rand_index]
        del m_parents[rand_index]
        already_selected += 1
        rand_index = random.randint(0, n_selection - already_selected - 1)
        parent2 = m_parents[rand_index]
        del m_parents[rand_index]
        already_selected += 1

        child_A, child_B = get_children(parent1, parent2)
        children.append(child_A)
        children.append(child_B)

    return children


def get_parents_turniej():
    # selekcja turniejowa
    parents_ret = []

    for _ in range(n_selection):
        rand_index1 = random.randint(0, len(population) - 1)
        rand_index2 = random.randint(0, len(population) - 1)
        parent1 = population[rand_index1]
        parent2 = population[rand_index2]

        if fitness(items, knapsack_max_capacity, parent1) > fitness(items, knapsack_max_capacity, parent1):
            parents_ret.append(parent1)
        else:
            parents_ret.append(parent2)

    return parents_ret


def reszta(myN):
    parents_ret = []

    for _ in range(myN):
        rand_index1 = random.randint(0, len(population) - 1)
        rand_index2 = random.randint(0, len(population) - 1)
        parent1 = population[rand_index1]
        parent2 = population[rand_index2]

        if fitness(items, knapsack_max_capacity, parent1) > fitness(items, knapsack_max_capacity, parent1):
            parents_ret.append(parent1)
        else:
            parents_ret.append(parent2)

    return parents_ret


def get_parents():
    parents = []

    for _ in range(n_selection):
        parents.append(select_one())
    return parents


def select_one():
    max = sum([fitness(items, knapsack_max_capacity, individual) for individual in population])
    selection_probs = [fitness(items, knapsack_max_capacity, ind) / max for ind in population]
    return population[npr.choice(len(population), p=selection_probs)]


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


items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 200
n_selection = 60
n_elite = 5

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []

# 1. initial population
population = initial_population(len(items), population_size)

for _ in range(generations):
    population_history.append(population)

    best_specimen = get_best()
    # TODO: implement genetic algorithm
    # 2
    parents = get_parents_turniej()
    # 3
    children = create_children(parents)
    # 4
    mutate_children(children)

    # ?

    population = children
    for elite in best_specimen:
        population.append(elite)
    rmeszta = reszta(population_size -len(population))

    for person in rmeszta:
        population.append(person)

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
