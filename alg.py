import numpy as np
from itertools import product
from multiprocessing import Pool
from functools import partial
import pandas as pd

def greedy_tsp(dist_matrix):
    n = len(dist_matrix)
    visited = [False] * n
    path = []
    
    current_city = 0
    path.append(current_city)
    visited[current_city] = True
    total_distance = 0
    
    for _ in range(n - 1):
        nearest_city = None
        min_distance = float('inf')
        
        # Find the nearest unvisited city
        for city in range(n):
            if not visited[city] and dist_matrix[current_city][city] < min_distance:
                nearest_city = city
                min_distance = dist_matrix[current_city][city]
        
        # Move to the nearest city
        path.append(nearest_city)
        visited[nearest_city] = True
        total_distance += min_distance
        current_city = nearest_city
    
    # Return to the start city
    total_distance += dist_matrix[current_city][path[0]]
    path.append(path[0])
    
    return path, total_distance

def input_matrix():
    n = int(input("Enter the size of the matrix: "))
    matrix = []
    
    print("Enter the matrix row by row (space-separated values):")
    for i in range(n):
        row = list(map(int, input(f"Row {i+1}: ").split()))
        if len(row) != n:
            raise ValueError("Each row must have exactly n elements.")
        matrix.append(row)
    
    return np.array(matrix)

def generate_random_dist_matrix(n):
    matrix = np.random.randint(1, 101, size=(n, n))
    np.fill_diagonal(matrix, 0)
    matrix = (matrix + matrix.T) // 2
    return matrix

def matrix_file(filename):
    coordinates = []
    with open(filename, 'r') as file:
        next(file)
        for line in file:
            _, x, y = map(int, line.strip().split())
            coordinates.append((x, y))

    coordinates = np.array(coordinates)
    diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    return distance_matrix

def generate_file(num_cities, filename="tsp_instance.txt", x_range=(0, 2000), y_range=(0, 2000)):
    x_coords = np.random.randint(x_range[0], x_range[1], num_cities)
    y_coords = np.random.randint(y_range[0], y_range[1], num_cities)

    with open(filename, 'w') as f:
        f.write(f"{num_cities}\n")
        for i in range(num_cities):
            f.write(f"{i + 1} {x_coords[i]} {y_coords[i]}\n")

    print(f"Generated TSP file with {num_cities} cities: {filename}")

def ant_colony_tsp(dist_matrix, num_ants=10, num_iterations=200, alpha=1.0, beta=5.0, evaporation_rate=0.5, seed=42):
    if seed is not None:
        np.random.seed(seed)
    
    n = len(dist_matrix)
    pheromone = np.ones((n, n)) / n
    best_path = None
    best_distance = float('inf')

    for _ in range(num_iterations):
        all_paths = []
        all_distances = []

        for _ in range(num_ants):
            path = [np.random.randint(n)]
            visited = set(path)

            while len(path) < n:
                current_city = path[-1]
                probabilities = []

                for city in range(n):
                    if city not in visited:
                        pheromone_level = pheromone[current_city][city] ** alpha
                        visibility = (1 / dist_matrix[current_city][city]) ** beta
                        probabilities.append(pheromone_level * visibility)
                    else:
                        probabilities.append(0)

                probabilities = np.array(probabilities)
                if probabilities.sum() == 0:
                    probabilities = np.ones(n) / n
                else:
                    probabilities /= probabilities.sum()
                next_city = np.random.choice(range(n), p=probabilities)
                path.append(next_city)
                visited.add(next_city)

            path.append(path[0])
            distance = sum(dist_matrix[path[i]][path[i + 1]] for i in range(n))
            all_paths.append(path)
            all_distances.append(distance)

            if distance < best_distance:
                best_path = path
                best_distance = distance

        pheromone *= (1 - evaporation_rate)
        for path, distance in zip(all_paths, all_distances):
            for i in range(n):
                pheromone[path[i]][path[i + 1]] += 1 / distance

    return best_path, best_distance

def evaluate(params, dist_matrix, seed=42):
    num_ants, num_iterations, alpha, beta, evaporation_rate = params
    path, distance = ant_colony_tsp(dist_matrix, num_ants, num_iterations, alpha, beta, evaporation_rate, seed=seed)
    return (params, distance)

def grid_search_aco(dist_matrix, num_ants_range=None, num_iterations_range=None, alpha_range=None, beta_range=None, evaporation_rate_range=None, seed=42):
    if num_ants_range is None:
        num_ants_range = [5, 10, 20, 30]
    if num_iterations_range is None:
        num_iterations_range = [50,100,200]
    if alpha_range is None:
        alpha_range = [0.5, 1.0, 1.5, 2.0]
    if beta_range is None:
        beta_range = [1.0, 2.0, 3.0, 4.0, 5.0]
    if evaporation_rate_range is None:
        evaporation_rate_range = [0.1, 0.3, 0.5, 0.7, 0.9]

    best_params = None
    best_distance = float('inf')

    param_combinations = list(product(num_ants_range, num_iterations_range, alpha_range, beta_range, evaporation_rate_range))
    
    with Pool() as pool:
        results = pool.map(partial(evaluate, dist_matrix=dist_matrix, seed=seed), param_combinations)
    results_df = pd.DataFrame(results, columns=['Parameters', 'Distance'])
    results_df.to_csv('aco_grid_search_results.csv', index=False)
    for params, distance in results:
        if distance < best_distance:
            best_distance = distance
            best_params = params

    return best_params, best_distance
