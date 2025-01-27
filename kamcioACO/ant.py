import numpy as np
import random
import timeit
import time
from itertools import product
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt

class ACO:
    def __init__(self, ant_number=None, iterations=None, pheromone_evaporation=None, alpha=None, beta=None, epsilon=None):
        self.ant_number = ant_number
        self.iterations = iterations
        self.pheromone_evaporation = pheromone_evaporation
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.matrix = None
        self.visibility = None
        self.pheromone = None
        self.best_path = None
        self.best_path_length = float('inf')
        self.best_path_history = []

    def init_matrix(self, coordinates_file, pheromone_start, visibility_const):
        with open(coordinates_file, 'r') as file:
            lines = file.readlines()
            num_cities = int(lines[0])
            coordinates = []
            for line in lines[1:num_cities + 1]:
                parts = line.strip().split()
                coordinates.append((int(parts[1]), int(parts[2])))

        dist_matrix = np.linalg.norm(
            np.array(coordinates)[:, None, :] - np.array(coordinates)[None, :, :], axis=-1
        )
        np.fill_diagonal(dist_matrix, np.inf)
        self.pheromone = np.full((num_cities, num_cities), pheromone_start)
        self.visibility = np.where(dist_matrix != np.inf, visibility_const / dist_matrix, 0)
        self.visibility /= self.visibility.sum(axis=1, keepdims=True)
        self.matrix = np.stack([self.visibility, self.pheromone, dist_matrix], axis=-1)

    def generate_path(self, start):
        num_cities = len(self.matrix)
        path = [start]
        visited = np.zeros(num_cities, dtype=bool)
        visited[start] = True
        current_city = start

        for _ in range(num_cities - 1):
            if random.random() < self.epsilon:
                next_city = np.random.choice(np.where(~visited)[0])
            else:
                pheromone_alpha = self.pheromone[current_city] ** self.alpha
                visibility_beta = self.visibility[current_city] ** self.beta

                probabilities = pheromone_alpha * visibility_beta
                probabilities[visited] = 0

                if probabilities.sum() == 0:
                    next_city = np.random.choice(np.where(~visited)[0])
                else:
                    probabilities /= probabilities.sum()
                    next_city = np.random.choice(num_cities, p=probabilities)

            path.append(next_city)
            visited[next_city] = True
            current_city = next_city

        path.append(start)
        return path

    def path_length(self, path):
        indices = np.array(path)
        lengths = self.matrix[indices[:-1], indices[1:], 2]
        return lengths.sum()

    def spread_pheromones(self, paths):
        path_lengths = [self.path_length(path) for path in paths]

        self.pheromone *= (1 - self.pheromone_evaporation)

        for path, length in zip(paths, path_lengths):
            for i, j in zip(path[:-1], path[1:]):
                self.pheromone[i][j] += 1 / length

        for i, j in zip(self.best_path[:-1], self.best_path[1:]):
            self.pheromone[i][j] += 10 / self.best_path_length

    def apply_local_pheromone_update(self, path):
        decay_factor = 0.05
        for i, j in zip(path[:-1], path[1:]):
            self.pheromone[i][j] *= (1 - decay_factor)
            self.pheromone[i][j] += decay_factor / self.best_path_length

    def greedy_initial_solution(self, start):
        num_cities = len(self.matrix)
        path = [start]
        visited = np.zeros(num_cities, dtype=bool)
        visited[start] = True
        current_city = start

        for _ in range(num_cities - 1):
            remaining = np.where(~visited)[0]
            next_city = remaining[np.argmin(self.matrix[current_city, remaining, 2])]
            path.append(next_city)
            visited[next_city] = True
            current_city = next_city

        path.append(start)
        return path

    def two_opt(self, path):
        best_path = np.array(path)
        best_length = self.path_length(best_path)

        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path) - 1):
                new_path = np.concatenate((best_path[:i], best_path[i:j + 1][::-1], best_path[j + 1:]))
                new_length = self.path_length(new_path)
                if new_length < best_length:
                    best_path = new_path
                    best_length = new_length
        return best_path.tolist()

    def three_opt(self, path):
        best_path = np.array(path)
        best_length = self.path_length(best_path)

        n = len(path) - 1
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    # Create all possible 3-opt moves
                    new_paths = [
                        np.concatenate((best_path[:i], best_path[i:j][::-1], best_path[j:k][::-1], best_path[k:])),
                        np.concatenate((best_path[:i], best_path[j:k], best_path[i:j], best_path[k:])),
                        np.concatenate((best_path[:i], best_path[j:k][::-1], best_path[i:j][::-1], best_path[k:])),
                    ]
                    # Find the best one
                    for new_path in new_paths:
                        new_length = self.path_length(new_path)
                        if new_length < best_length:
                            best_path = new_path
                            best_length = new_length
        return best_path.tolist()


    def run(self):
        # fixed_start = 0
        # initial_path = self.greedy_initial_solution(fixed_start)
        # initial_path_length = self.path_length(initial_path)
        # self.best_path = initial_path
        # self.best_path_length = self.path_length(initial_path)

        num_cities = len(self.matrix)
        starting_cities = list(range(num_cities))
        random.shuffle(starting_cities)
        start_time = time.time()
        time_limit = 170

        for iteration in range(self.iterations):
            if time.time() - start_time > time_limit:
                print(f"Stopped after {iteration} iterations due to time limit")
                break
            self.epsilon = max(0.01, self.epsilon * 0.95)

            paths = [self.generate_path(starting_cities[i % num_cities]) for i in range(self.ant_number)]

            for path in paths:
                # self.apply_local_pheromone_update(path)
                path_length = self.path_length(path)
                if path_length < self.best_path_length:
                    self.best_path = path
                    self.best_path_length = path_length

            self.spread_pheromones(paths)
            self.best_path_history.append(self.best_path_length)
            self.pheromone_evaporation = max(0.1, self.pheromone_evaporation * 0.99)
            self.best_path = self.two_opt(self.best_path)
            self.best_path_length = self.path_length(self.best_path)
            print(f"Iteration {iteration + 1}/{self.iterations}, Best Path Length: {self.best_path_length}")
            # print("Best Path:", ",".join(map(str, self.best_path)))

        # print(f"Initial Path Length: {initial_path_length}")
        # print("Best Path:", ",".join(map(str, self.best_path)))
        # print("Best Path History:", self.best_path_history)
        # plt.plot(self.best_path_history)
        # plt.xlabel('Iteration')
        # plt.ylabel('Best Path Length')
        # plt.title('ACO Optimization Progress')
        # plt.show()

        return {
            "ant_number": self.ant_number,
            "iterations": self.iterations,
            "pheromone_evaporation": self.pheromone_evaporation,
            "alpha": self.alpha,
            "beta": self.beta,
            "epsilon": self.epsilon,
            "best_path": " -> ".join(str(p) for p in self.best_path),
            "best_path_length": self.best_path_length
        }

# def run_time():
#     test = ACO(ant_number=200, iterations=40, pheromone_evaporation=0.2, alpha=1, beta=5.7, epsilon=0.04)
#     test.init_matrix('data/benchmark_instances_transformed/transformed_st70.tsp',  pheromone_start=0.001, visibility_const=200)
#     return test.run()
#
# execution_time = timeit.timeit(run_time, number=1)
# print(f"Execution time: {execution_time} seconds")

def evaluate(params, coordinates_file, pheromone_start, visibility_const, seed=42):
    num_ants, num_iterations, alpha, beta, evaporation_rate, epsilon = params
    aco = ACO(ant_number=num_ants, iterations=num_iterations, pheromone_evaporation=evaporation_rate, alpha=alpha, beta=beta, epsilon=epsilon)
    aco.init_matrix(coordinates_file, pheromone_start, visibility_const)
    result = aco.run()
    return (params, result['best_path_length'])

def grid_search_aco(coordinates_file, pheromone_start, visibility_const, num_ants_range=None, num_iterations_range=None, alpha_range=None, beta_range=None, evaporation_rate_range=None, epsilon_range=None, seed=42):
    if num_ants_range is None:
        num_ants_range = [100]
    if num_iterations_range is None:
        num_iterations_range = [30]
    if alpha_range is None:
        alpha_range = [1,2,3,4,5]
    if beta_range is None:
        beta_range = [1,2,3,4,5]
    if evaporation_rate_range is None:
        evaporation_rate_range = [0.1,0.2,0.3,0.4,0.5]
    if epsilon_range is None:
        epsilon_range = [0.04]

    best_params = None
    best_distance = float('inf')

    param_combinations = list(product(num_ants_range, num_iterations_range, alpha_range, beta_range, evaporation_rate_range, epsilon_range))

    with Pool(processes=7) as pool:
        results = pool.map(partial(evaluate, coordinates_file=coordinates_file, pheromone_start=pheromone_start, visibility_const=visibility_const, seed=seed), param_combinations)

    for params, distance in results:
        if distance < best_distance:
            best_distance = distance
            best_params = params

    return best_params, best_distance

best_params, best_distance = grid_search_aco('data/tsp500.txt', pheromone_start=0.001, visibility_const=200)
print(best_params, best_distance)