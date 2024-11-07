import argparse
from timeit import default_timer as timer
import alg

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    while True:
        command = get_command()
        if command == 'exit':
            break
        if command == 'tsp_manual':
            dist_matrix = alg.input_matrix()
            print("Input matrix:")
            print(dist_matrix)
            start = timer()
            path, distance = alg.greedy_tsp(dist_matrix)
            end = timer()
            print("Shortest path:", path)
            print("Total distance:", distance)
            print("Execution time:", end - start, "seconds")
        if command == 'tsp_random':
            n = int(input("Enter the size of the matrix: "))
            dist_matrix = alg.generate_random_dist_matrix(n)
            print("Randomly generated matrix:")
            print(dist_matrix)
            start = timer()
            path, distance = alg.greedy_tsp(dist_matrix)
            end = timer()
            print("Shortest path:", path)
            print("Total distance:", distance)
            print("Execution time:", end - start, "seconds")
        if command == 'tsp_file':
            file_name = input("Enter file name: ")
            dist_matrix = alg.matrix_file(file_name)
            print("Input matrix:")
            print(dist_matrix)
            start = timer()
            path, distance = alg.greedy_tsp(dist_matrix)
            end = timer()
            print("Shortest path:", path)
            print("Total distance:", distance)
            print("Execution time:", end - start, "seconds")
        if command == 'generate':
            num_cities = int(input("Enter number of cities: "))
            file_name = input("Enter file name: ")
            range_x = tuple(map(int, input("Enter range for x coordinates (min, max): ").split()))
            range_y = tuple(map(int, input("Enter range for y coordinates (min, max): ").split()))
            alg.generate_file(num_cities, file_name, range_x, range_y)
        if command == 'generate_fast':
            num_cities = int(input("Enter number of cities: "))
            file_name = input("Enter file name: ")
            alg.generate_file(num_cities, file_name)
        if command == 'tsp_aco':
            file_name = input("Enter file name: ")
            dist_matrix = alg.matrix_file(file_name)
            #print("Input matrix:")
            #print(dist_matrix)
            start = timer()
            path, distance = alg.ant_colony_tsp(dist_matrix, num_ants=30, num_iterations=200, alpha=1.0, beta=2.0, evaporation_rate=0.3, seed=42)
            end = timer()
            print("Shortest path:", path)
            print("Total distance:", distance)
            print("Execution time:", end - start, "seconds")
        if command == 'grid_search_aco':
            file_name = input("Enter file name: ")
            dist_matrix = alg.matrix_file(file_name)
            print("Input matrix:")
            print(dist_matrix)
            num_ants_range = list(map(int, input("Enter range for num_ants (space-separated): ").split()))
            num_iterations_range = list(map(int, input("Enter range for num_iterations (space-separated): ").split()))
            alpha_range = list(map(float, input("Enter range for alpha (space-separated): ").split()))
            beta_range = list(map(float, input("Enter range for beta (space-separated): ").split()))
            evaporation_rate_range = list(map(float, input("Enter range for evaporation_rate (space-separated): ").split()))
            start = timer()
            best_params, best_distance = alg.grid_search_aco(dist_matrix, num_ants_range, num_iterations_range, alpha_range, beta_range, evaporation_rate_range, seed=42)
            end = timer()
            print("Best parameters:", best_params)
            print("Best distance:", best_distance)
            print("Execution time:", end - start, "seconds")
        if command == 'grid':
            file_name = input("Enter file name: ")
            dist_matrix = alg.matrix_file(file_name)
            start = timer()
            best_params, best_distance = alg.grid_search_aco(dist_matrix, seed=42)
            end = timer()
            print("Best parameters:", best_params)
            print("Best distance:", best_distance)
            print("Execution time:", end - start, "seconds)")
            
        elif command == 'help':
            print_help()
        else:
            print('Invalid command. Type "help" for a list of commands.')

def print_help():
    print('--- Help ---')
    print('tsp_manual - solve the TSP problem with a manually entered distance matrix')
    print('tsp_random - solve the TSP problem with a randomly generated distance matrix')
    print('tsp_file - solve the TSP problem with a distance matrix from a file')
    print('generate - generate a TSP instance file with random coordinates')
    print('generate_fast - generate a TSP instance file with random coordinates (default range)')
    print('tsp_aco - solve the TSP problem using Ant Colony Optimization')
    print('grid_search_aco - perform a grid search to find the best parameters for ACO')
    print('help - show this help message')
    print('exit - exit the program')

def get_command():
    return input('command> ').lower()

if __name__ == "__main__":
    main()