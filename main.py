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
            alg.generate_file(num_cities, file_name,range_x, range_y)
        if command == 'generate_fast':
            num_cities = int(input("Enter number of cities: "))
            file_name = input("Enter file name: ")
            alg.generate_file(num_cities, file_name)
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
    print('help - show this help message')
    print('exit - exit the program')

def get_command():
    return input('command> ').lower()

if __name__ == "__main__":
    main()
