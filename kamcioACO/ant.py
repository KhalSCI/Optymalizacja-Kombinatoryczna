import numpy as np
import random

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
        self.visibility = None
        self.pheromone_one = None
        self.pheromone_two = None
        self.cost_matrix = None

        self.best_path = None
        self.best_path_length = float('inf')
        self.best_path_history = []

    def init_matrix(self, file_name, pheromone_one_start, pheromone_two_start, visibility_const):
        with open(file_name, 'r') as file:
            lines = file.readlines()
            num_cities = int(lines[0])
            coordinates = []
            for line in lines[1:num_cities + 1]:
                parts = line.strip().split()
                coordinates.append((int(parts[1]), int(parts[2])))

        dist_matrix = np.linalg.norm(
            np.array(coordinates)[:, None, :] - np.array(coordinates)[None, :, :], axis=-1
        )
        np.fill_diagonal(dist_matrix, np.inf)  # Nie można podróżować do siebie samego

        self.pheromone_one = np.full((num_cities, num_cities), pheromone_one_start)
        self.pheromone_two = np.full((num_cities, num_cities), pheromone_two_start)
        self.visibility = np.where(dist_matrix != np.inf, visibility_const / dist_matrix, 0)

        self.cost_matrix = dist_matrix * np.random.uniform(0.8, 1.1, size=dist_matrix.shape)
        self.matrix = np.stack([self.visibility, self.pheromone_one, self.pheromone_two, dist_matrix, self.cost_matrix], axis=-1)

    def generate_path(self, start):
        num_cities = len(self.matrix)
        path = [start]
        visited = np.zeros(num_cities, dtype=bool)
        visited[start] = True

        current_city = start

        for _ in range(num_cities - 1):
            pheromone_alpha = self.pheromone_one[current_city] ** self.alpha_one
            pheromone_beta = self.pheromone_two[current_city] ** self.alpha_two
            visibility_beta = self.visibility[current_city] ** self.beta

            probabilities = pheromone_alpha * pheromone_beta * visibility_beta
            probabilities[visited] = 0  # Wyklucz odwiedzone miasta

            if probabilities.sum() == 0:  # Obsługa przypadku, gdy brak prawdopodobieństwa
                next_city = np.random.choice(np.where(~visited)[0])
            else:
                probabilities /= probabilities.sum()
                next_city = np.random.choice(num_cities, p=probabilities)

            path.append(next_city)
            visited[next_city] = True
            current_city = next_city

        path.append(start)  # Wracamy do miasta startowego
        return path

    def path_metrics(self, path):
        indices = np.array(path)
        lengths = self.matrix[indices[:-1], indices[1:], 3]
        costs = self.matrix[indices[:-1], indices[1:], 4]
        return lengths.sum(), costs.sum()

    def spread_pheromones(self, paths):
        path_lengths = [self.path_metrics(path)[0] for path in paths]
        path_costs = [self.path_metrics(path)[1] for path in paths]

        self.pheromone_one *= (1 - self.pheromone_evaporation)
        self.pheromone_two *= (1 - self.pheromone_evaporation)

        for path, length, cost in zip(paths, path_lengths, path_costs):
            for i, j in zip(path[:-1], path[1:]):
                self.pheromone_one[i][j] += 1 / length
                self.pheromone_two[i][j] += 1 / cost

    def run(self):
        for iteration in range(self.iterations):
            paths = [self.generate_path(random.randint(0, len(self.matrix) - 1)) for _ in range(self.ant_number)]
            for path in paths:
                path_length, _ = self.path_metrics(path)
                if path_length < self.best_path_length:
                    self.best_path = path
                    self.best_path_length = path_length
            self.spread_pheromones(paths)
            self.best_path_history.append(self.best_path_length)

            print(f"Iteration {iteration + 1}/{self.iterations}, Best Path Length: {self.best_path_length}")

        print("Best Path:", " -> ".join(map(str, self.best_path)))
        print("Best Path History:", self.best_path_history)


        return {
            "ant_number": self.ant_number,
            "iterations": self.iterations,
            "pheromone_evaporation": self.pheromone_evaporation,
            "alpha_one": self.alpha_one,
            "alpha_two": self.alpha_two,
            "beta": self.beta,
            "epsilon": self.epsilon,
            "best_path": " -> ".join(str(p) for p in self.best_path),
            "best_path_length": self.best_path_length
        }

#
# test = ACO(ant_number=65, iterations=101, pheromone_evaporation=0.3974180, alpha_one=2.7731535401984324, alpha_two=2.5, beta=4.066823531422358, epsilon=0.1)
# test.init_matrix('./data/berlin/berlin52.txt', pheromone_one_start=0.200, pheromone_two_start=0.200, visibility_const=200)
# test.run()
