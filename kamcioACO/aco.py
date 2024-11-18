import numpy as np
import random
import csv

import pandas as pd


class ACO:
    def __init__(self, ant_number=None, iterations=None, pheromone_evaporation=None, alpha=None, beta=None):
        self.matrix = None
        self.cities = None
        self.ant_number = ant_number
        self.iterations = iterations
        self.pheromone_evaporation = pheromone_evaporation
        self.alpha = alpha
        self.beta = beta
        self.pheromone = None
        self.visibility = None
        self.probabilities = None
        self.best_path = None
        self.best_path_length = None
        self.best_path_history = None

    def init_matrix(self, file_name, pheromones_start, visibility_const):
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

        self.pheromone = np.full((num_cities, num_cities), pheromones_start)
        self.visibility = np.zeros((num_cities, num_cities))

        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    self.visibility[i][j] = visibility_const / dist_matrix[i][j]

        self.matrix = np.empty((num_cities, num_cities), dtype=object)
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    self.matrix[i][j] = [self.visibility[i][j], self.pheromone[i][j], dist_matrix[i][j]]
                else:
                    self.matrix[i][j] = [0, self.pheromone[i][j], 0]

    def print_matrix(self):
        if self.matrix is not None:
            for row in self.matrix:
                print(" ".join(f"({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})" for v in row))
        else:
            print("Matrix is not initialized.")

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
        visited = set()
        visited.add(start)
        prev = start
        cities = len(self.matrix)
        for i in range(cities - 1):
            probabilities = self.get_probabilities(prev, visited)
            probabilities = [(p[0] ** self.alpha) * (p[1] ** self.beta) for p in probabilities]
            probabilities = [p / sum(probabilities) for p in probabilities]
            next_city = np.random.choice(cities, p=probabilities)
            path.append(next_city)
            visited.add(next_city)
            prev = next_city
            if len(visited) == cities:
                visited.clear()
                visited.add(next_city)
        path.append(start)
        return path

    def generate_paths(self):
        paths = []
        for i in range(self.ant_number):
            start_city = i % len(self.matrix)
            paths.append(self.generate_path(start_city))
        return paths

    def path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += self.matrix[path[i]][path[i + 1]][2]
        return length

    def print_path(self, path):
        print(" -> ".join(str(p) for p in path))
        print(f" \n Path length: {self.path_length(path)}")

    def spread_pheromones(self, paths):
        # Obliczenie długości każdej ścieżki
        path_lengths = [(path, self.path_length(path)) for path in paths]

        sorted_paths = sorted(path_lengths, key=lambda x: x[1])

        median_length = np.median([length for _, length in sorted_paths])
        top_half = [path for path, length in sorted_paths if length < median_length]
        top_two = sorted_paths[:2]

        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                self.matrix[i][j][1] *= (1 - self.pheromone_evaporation)

        for path, length in sorted_paths:
            for i in range(len(path) - 1):
                self.matrix[path[i]][path[i + 1]][1] += 1 / length
            self.matrix[path[-1]][path[0]][1] += 1 / length

        for path in top_half:
            for i in range(len(path) - 1):
                self.matrix[path[i]][path[i + 1]][1] += 1 / self.path_length(path)
            self.matrix[path[-1]][path[0]][1] += 1 / self.path_length(path)

        for path, length in top_two:
           for i in range(len(path) - 1):
               self.matrix[path[i]][path[i + 1]][1] += 1 / length
           self.matrix[path[-1]][path[0]][1] += 1 / length

    def ant_queen(self):
        pass

    def run(self):
        self.best_path = None
        self.best_path_length = float('inf')
        self.best_path_history = []

        for i in range(self.iterations):
            paths = self.generate_paths()
            for path in paths:
                length = self.path_length(path)
                if length < self.best_path_length:
                    self.best_path = path
                    self.best_path_length = length
            self.best_path_history.append(self.best_path_length)
            self.spread_pheromones(paths)
            print(f"Iteration {i + 1}/{self.iterations}")

        print("Best path:")
        self.print_path(self.best_path)
        print("Best path history:")
        print(self.best_path_history)


        return {
            "ant_number": self.ant_number,
            "iterations": self.iterations,
            "pheromone_evaporation": self.pheromone_evaporation,
            "alpha": self.alpha,
            "beta": self.beta,
            "best_path": " -> ".join(str(p) for p in self.best_path),
            "best_path_length": self.best_path_length
        }

def run_aco(ant_number, iterations, pheromone_evaporation, alpha, beta):
    aco = ACO(ant_number=ant_number, iterations=iterations, pheromone_evaporation=pheromone_evaporation, alpha=alpha, beta=beta)
    aco.init_matrix("data/bier/bier127.txt", pheromones_start=0.1000, visibility_const=400)
    return aco.run()

def analyze_results(df):
    if df.empty:
        return {
            'ant_number': 127,
            'iterations': 30,
            'pheromone_evaporation': 0.5,
            'alpha': 1.5,
            'beta': 3.5
        }

    best_result = df['result'].min()
    best_params = df[df['result'] == best_result].iloc[0]

    return {
        'ant_number': int(best_params['ant_number']),
        'iterations': int(best_params['iterations']),
        'pheromone_evaporation': float(best_params['pheromone_evaporation']),
        'alpha': float(best_params['alpha']),
        'beta': float(best_params['beta'])
    }


def adjust_parameters(best_params):
    ant_number = best_params['ant_number'] + random.randint(-7, 12)
    iterations = best_params['iterations'] + random.randint(-7, 12)
    pheromone_evaporation = best_params['pheromone_evaporation'] + random.uniform(-0.02000, 0.02000)
    alpha = best_params['alpha'] + random.uniform(-0.04000000, 0.0400000)
    beta = best_params['beta'] + random.uniform(-0.040000000, 0.04000000)

    ant_number = max(135, ant_number)
    iterations = max(20, iterations)
    pheromone_evaporation = min(max(0.1, pheromone_evaporation), 0.9)
    alpha = min(max(0.5, alpha), 3.5)
    beta = min(max(2.0, beta), 5.0)

    return ant_number, iterations, pheromone_evaporation, alpha, beta


from multiprocessing import Pool

def run_aco_helper(params):
    ant_number, iterations, pheromone_evaporation, alpha, beta = params
    return run_aco(ant_number, iterations, pheromone_evaporation, alpha, beta)

def run_multiple():
    results = []

    try:
        df = pd.read_csv('data/bier/bier127.csv',
                         names=['ant_number', 'iterations', 'pheromone_evaporation', 'alpha', 'beta', 'path', 'result'])
        best_params = analyze_results(df)
    except FileNotFoundError:
        best_params = {
            'ant_number': 127,
            'iterations': 30,
            'pheromone_evaporation': 0.5,
            'alpha': 1.5,
            'beta': 3.5
        }

    params_list = [adjust_parameters(best_params) for _ in range(15)]

    with Pool(processes=5) as pool:
        results = pool.map(run_aco_helper, params_list)

    df = pd.DataFrame(results)
    df.to_csv('data/bier/bier127.csv', mode='a', header=False, index=False)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    run_multiple()

# TODO:
#   można dodac parametr epsilon który jest współczynnikiem eksploracji, pozwala na odblokowanie nowych ścieżek jak algorytm sie zablokował w lokalnym optimum
#   implementacja -> self.epsilon = epsilon i podajemy jego wartoc w konstruktorze
#   for i in range(cities - 1):
#               # Pobierz prawdopodobieństwa wyboru kolejnego miasta
#               probabilities = self.get_probabilities(prev, visited)
#               probabilities = [(p[0] ** self.alpha) * (p[1] ** self.beta) for p in probabilities]
#               probabilities = [p / sum(probabilities) for p in probabilities]  # normalizacja
#               # Zastosowanie polityki ε-greedy
#               if np.random.rand() < self.epsilon:
#                   # Eksploracja: wybierz losowe nieodwiedzone miasto
#                   unvisited = [idx for idx in range(cities) if idx not in visited]
#                   next_city = np.random.choice(unvisited)
#               else:
#                   # Eksploatacja: wybierz miasto na podstawie prawdopodobieństw
#                   next_city = np.random.choice(cities, p=probabilities)
#               # Zaktualizuj ścieżkę i odwiedzone miasta
#               path.append(next_city)
#               visited.add(next_city)
#               prev = next_city
#   Można dodać rozszerzona wersje tego wyżej, nazwaną Levy Flight, SNK referencja pogu
#   https://link.springer.com/article/10.1007/s40747-020-00138-3
#   Polega to mniej weicej na tym że zamiast wybierac miasto z prawdopodobieństwem to wybieramy miasto z rozkładu (jednostajnego) Levy'ego
#   https://en.wikipedia.org/wiki/L%C3%A9vy_flight
#   https://en.wikipedia.org/wiki/L%C3%A9vy_distribution
#   Lot Levy'ego to metoda modelowania przypadkowych trajektorii, które są długimi skokami (eksploracja) przeplatanymi krótszymi ruchami (eksploatacja).
#   W kontekście ACO stosuje się ją do generowania lepszych ścieżek poprzez wprowadzenie losowości w eksploracji przestrzeni rozwiązań
