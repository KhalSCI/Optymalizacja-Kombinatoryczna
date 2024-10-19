import numpy as np

def greedy_tsp(dist_matrix):
    n = len(dist_matrix)
    visited = [False] * n
    path = []
    
    # Start from the first city (0)
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
    path.append(path[0])  # Complete the cycle
    
    return path, total_distance


dist_matrix = np.array([[0, 10, 15, 20, 25],
                        [10, 0, 35, 25, 30],
                        [15, 35, 0, 30, 20],
                        [20, 25, 30, 0, 15],
                        [25, 30, 20, 15, 0]])
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
import numpy as np

def generate_random_dist_matrix(n):
    
    matrix = np.random.randint(1, 101, size=(n, n))
    
    
    np.fill_diagonal(matrix, 0)
    
    
    matrix = (matrix + matrix.T) // 2
    
    return matrix
