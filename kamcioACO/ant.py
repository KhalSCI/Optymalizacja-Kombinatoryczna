import numpy as np
import random
import csv
import pandas as pd

class ACO:
    def __init__(self, ant_number=None, iterations=None, pheromone_evaporation=None, alpha_one=None, alpha_two=None, beta=None, epsilon=None):
        self.ant_number = ant_number
        self.iterations = iterations
        self.pheromone_evaporation = pheromone_evaporation
        self.alpha_one = alpha_one
        self.alpha_two = alpha_two
        self.beta = beta
        self.epsilon = epsilon

        self.matrix = None
        self.cities = None
        self.visibility = None
        self.probabilities = None

        self.pheromone_one = None
        self.pheromone_two = None
        self.cost_matrix = None

        self.best_path = None
        self.best_cost = None
        self.best_path_history = None

    def init_matrix(self, file_name, pheromone_one_start, pheromone_two_start, visibility_const):
        with open(file_name, 'r') as file:
            lines = file.readlines()
            num_cities = int(lines[0])
            coordinates = []
            dist_matrix = np.zeros((num_cities, num_cities))
            for line in lines[1:num_cities + 1]:
                parts = line.strip().split()
                coordinates.append((int(parts[1]), int(parts[2])))
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    dist_matrix[i][j] = np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j]))
        self.pheromone_two = np.full((num_cities, num_cities), pheromone_two_start)
        self.pheromone_one = np.full((num_cities, num_cities), pheromone_one_start)

        self.visibility = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    self.visibility[i][j] = visibility_const / dist_matrix[i][j]
        self.cost_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    self.cost_matrix[i][j] = dist_matrix[i][j] * random.uniform(0.8, 1.1)
        self.matrix = np.empty((num_cities, num_cities), dtype=object)
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    self.matrix[i][j] = [self.visibility[i][j], self.pheromone_one[i][j], self.pheromone_two[i][j] , dist_matrix[i][j], self.cost_matrix[i][j]]
                else:
                    self.matrix[i][j] = [0, self.pheromone_one[i][j], self.pheromone_two[i][j], 0, 0]



    def get_probabilities(self, city, visited):
        probabilities = []
        for i in range(len(self.matrix)):
            if i not in visited:
                probabilities.append(self.matrix[city][i])
            else:
                probabilities.append((0, 0))
        return probabilities

    def generate_path(self, start):
        path = [start]
        visited = {start}
        prev = start
        cities = len(self.matrix)

        while len(visited) < cities:
            probabilities = []
            for i in range(cities):
                if i not in visited:
                    pheromone_one = self.matrix[prev][i][1]
                    pheromone_two = self.matrix[prev][i][2]
                    visibility = self.matrix[prev][i][0]
                    combined_pheromone = ((pheromone_one ** self.alpha_one) * (pheromone_two ** self.alpha_two))
                    cost_factor = 1 / self.cost_matrix[prev][i]
                    probabilities.append(combined_pheromone * (visibility ** self.beta) * cost_factor)
                else:
                    probabilities.append(0)
            total_probability= sum(probabilities)

            if total_probability == 0:
                probabilities = [1 if i not in visited else 0 for i in range(cities)]
                total_probability = sum(probabilities)
            probabilities = [p / total_probability for p in probabilities]

        # tutdaj mozna dodac epsilon widoczny w kodzie aco.py


            next_city = np.random.choice(cities, p=probabilities)
            path.append(next_city)
            visited.add(next_city)
            prev = next_city
        return path

    def generate_paths(self):
        paths = []
        for i in range(len(self.matrix)):
            paths.append(self.generate_path(i))
        return paths

    def path_metrics(self, path):
        length = 0
        cost = 0
        for i in range(len(path) - 1):
            length += self.matrix[path[i]][path[i + 1]][3]
            cost += self.matrix[path[i]][path[i + 1]][4]
        return length, cost

    def spread_pheromones(self, paths):
        length, cost = self.path_metrics()
