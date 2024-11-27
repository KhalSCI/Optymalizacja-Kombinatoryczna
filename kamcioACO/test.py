import numpy as np
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pandas as pd


class ACO:
    def __init__(self, ant_number=None, iterations=None, pheromone_evaporation=None, alpha=None, beta=None, epsilon=None):
        self.matrix = None
        self.cities = None
        self.ant_number = ant_number
        self.iterations = iterations
        self.pheromone_evaporation = pheromone_evaporation
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.exploration_decay = 0.95
        self.pheromone = None
        self.visibility = None
        self.best_path = None
        self.best_path_length = None
        self.best_path_history = None
        self.coordinates = None

    def init_matrix(self, file_name, pheromones_start, visibility_const):
        with open(file_name, 'r') as file:
            lines = file.readlines()
            num_cities = int(lines[0])
            self.coordinates = []
            dist_matrix = np.zeros((num_cities, num_cities))

            for line in lines[1:num_cities + 1]:
                parts = line.split()
                self.coordinates.append((float(parts[1]), float(parts[2])))

            for i in range(num_cities):
                for j in range(num_cities):
                    if i != j:
                        dist_matrix[i][j] = np.linalg.norm(np.array(self.coordinates[i]) - np.array(self.coordinates[j]))

        min_dist = np.min(dist_matrix)
        if min_dist == 0:
            min_dist = 1  # or some other small value to avoid division by zero
        self.pheromone = np.full((num_cities, num_cities), 1 / (num_cities * min_dist))
        self.visibility = np.zeros((num_cities, num_cities))

        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    self.visibility[i][j] = visibility_const / dist_matrix[i][j]

        self.matrix = np.empty((num_cities, num_cities), dtype=object)
        for i in range(num_cities):
            for j in range(num_cities):
                self.matrix[i][j] = (self.visibility[i][j], self.pheromone[i][j], dist_matrix[i][j])

    def generate_path(self, start):
        path = [start]
        visited = {start}
        prev = start
        cities = len(self.matrix)

        while len(visited) < cities:
            probabilities = []
            for i in range(cities):
                if i not in visited:
                    pheromone = self.matrix[prev][i][1]
                    visibility = self.matrix[prev][i][0]
                    probabilities.append((pheromone ** self.alpha) * (visibility ** self.beta))
                else:
                    probabilities.append(0)

            total_probability = sum(probabilities)
            if total_probability == 0:
                probabilities = [1 / cities] * cities  # or some other default probabilities
            else:
                probabilities = [p / total_probability for p in probabilities]

            if np.random.rand() < self.epsilon * (self.exploration_decay ** len(visited)):
                next_city = random.choice([i for i in range(cities) if i not in visited])
            else:
                next_city = np.random.choice(cities, p=probabilities)

            path.append(next_city)
            visited.add(next_city)
            prev = next_city

        path.append(start)
        return path

    def path_length(self, path):
        return sum(self.matrix[path[i]][path[i + 1]][2] for i in range(len(path) - 1))

    def spread_pheromones(self, paths):
        path_lengths = [(path, self.path_length(path)) for path in paths]
        sorted_paths = sorted(path_lengths, key=lambda x: x[1])
        self.pheromone *= (1 - self.pheromone_evaporation)

        for path, length in sorted_paths[:2]:
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i + 1]] += 1 / length
            self.pheromone[path[-1]][path[0]] += 1 / length

        if self.best_path:
            for i in range(len(self.best_path) - 1):
                self.pheromone[self.best_path[i]][self.best_path[i + 1]] += 1 / self.best_path_length
            self.pheromone[self.best_path[-1]][self.best_path[0]] += 1 / self.best_path_length

    def run(self):
        self.best_path = None
        self.best_path_length = float('inf')
        self.best_path_history = []

        for i in range(self.iterations):
            self.alpha = 1 + (self.alpha - 1) * (1 - i / self.iterations)
            self.beta = 2 + (self.beta - 2) * (i / self.iterations)

            paths = [self.generate_path(start % len(self.matrix)) for start in range(self.ant_number)]

            for path in paths:
                length = self.path_length(path)
                if length < self.best_path_length:
                    self.best_path = path
                    self.best_path_length = length
            self.best_path_history.append(self.best_path_length)
            self.spread_pheromones(paths)

            self.epsilon = max(self.epsilon * 0.98, 0.01)
            print(f"Iteration {i + 1}/{self.iterations} - Best Path Length: {self.best_path_length}")

        return {
            "ant_number": self.ant_number,
            "iterations": self.iterations,
            "pheromone_evaporation": self.pheromone_evaporation,
            "alpha": self.alpha,
            "beta": self.beta,
            "epsilon": self.epsilon,
            "best_path": self.best_path,
            "best_path_length": self.best_path_length
        }

    def visualize_path(self):
        if self.coordinates and self.best_path:
            x = [self.coordinates[city][0] for city in self.best_path]
            y = [self.coordinates[city][1] for city in self.best_path]
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, marker='o')
            plt.title("Best Path")
            plt.show()

    def visualize_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_path_history)
        plt.title("Best Path Length Over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Path Length")
        plt.show()


def analyze_results(df):
    if df.empty or 'result' not in df.columns:
        return {
            'ant_number': 52,
            'iterations': 60,
            'pheromone_evaporation': 0.3,
            'alpha': 1.5,
            'beta': 3.5,
            'epsilon': 0.1
        }

    df['weighted_result'] = df['result'] * (0.95 ** np.arange(len(df))[::-1])
    best_result = df['weighted_result'].min()
    best_params = df[df['weighted_result'] == best_result].iloc[0]

    return {
        'ant_number': int(best_params['ant_number']),
        'iterations': int(best_params['iterations']),
        'pheromone_evaporation': float(best_params['pheromone_evaporation']),
        'alpha': float(best_params['alpha']),
        'beta': float(best_params['beta']),
        'epsilon': float(best_params['epsilon'])
    }


def run_aco(ant_number, iterations, pheromone_evaporation, alpha, beta, epsilon):
    aco = ACO(ant_number=ant_number, iterations=iterations,
              pheromone_evaporation=pheromone_evaporation, alpha=alpha, beta=beta, epsilon=epsilon)
    aco.init_matrix('data/berlin/berlin52.txt', pheromones_start=0.2, visibility_const=200)
    results = aco.run()
    return results


def adjust_parameters(best_params):
    ant_number = best_params['ant_number'] + random.randint(-5, 8)
    iterations = best_params['iterations'] + random.randint(-5, 15)
    pheromone_evaporation = best_params['pheromone_evaporation'] + random.uniform(-0.201, 0.201)
    alpha = best_params['alpha'] + random.uniform(-0.205, 0.205)
    beta = best_params['beta'] + random.uniform(-0.205, 0.205)
    epsilon = best_params['epsilon'] + random.uniform(-0.02, 0.02)

    ant_number = max(70, ant_number)
    iterations = max(20, iterations)
    pheromone_evaporation = min(max(0.1, pheromone_evaporation), 0.9)
    alpha = min(max(0.5, alpha), 4.5)
    beta = min(max(2.0, beta), 7)
    epsilon = min(max(0.01, epsilon), 0.4)

    return ant_number, iterations, pheromone_evaporation, alpha, beta, epsilon


def run_aco_helper(params):
    ant_number, iterations, pheromone_evaporation, alpha, beta, epsilon = params
    return run_aco(ant_number, iterations, pheromone_evaporation, alpha, beta, epsilon)


def run_multiple():
    results = []

    try:
        df = pd.read_csv('b52.csv', delimiter='a')
    except FileNotFoundError:
        df = pd.DataFrame()

    best_params = analyze_results(df)
    initial_params = (best_params['ant_number'], best_params['iterations'],
                      best_params['pheromone_evaporation'], best_params['alpha'],
                      best_params['beta'], best_params['epsilon'])

    params_list = [adjust_parameters(best_params) for _ in range(6)]
    params_list.insert(0, initial_params)

    with Pool(6) as p:
        outputs = p.map(run_aco_helper, params_list)

    for params, output in zip(params_list, outputs):
        row = {
            'ant_number': params[0],
            'iterations': params[1],
            'pheromone_evaporation': params[2],
            'alpha': params[3],
            'beta': params[4],
            'epsilon': params[5],
            'result': output['best_path_length']
        }
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv('b52.csv', mode='a', header=False, index=False)


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    run_multiple()