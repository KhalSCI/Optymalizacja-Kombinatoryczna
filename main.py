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
        elif command == 'help':
            print_help()
        else:
            print('Invalid command. Type "help" for a list of commands.')

def print_help():
    print('--- Help ---')
    print('tsp_manual - solve the TSP problem with a manually entered distance matrix')
    print('tsp_random - solve the TSP problem with a randomly generated distance matrix')
    print('help - show this help message')
    print('exit - exit the program')

def get_command():
    return input('command> ').lower()

if __name__ == "__main__":
    main()
