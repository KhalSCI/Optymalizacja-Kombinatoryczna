import numpy as np

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
def generate_random_dist_matrix(n):
    
    matrix = np.random.randint(1, 101, size=(n, n))
    
    
    np.fill_diagonal(matrix, 0)
    
    
    matrix = (matrix + matrix.T) // 2
    
    return matrix

def matrix_file(filename):
    coordinates = []
    with open(filename, 'r') as file:
        # Skip the first line containing only the count of coordinates
        next(file)
        for line in file:
            _, x, y = map(int, line.strip().split())
            coordinates.append((x, y))

    # Convert the list of coordinates to a NumPy array
    coordinates = np.array(coordinates)

    # Step 2: Compute the Distance Matrix using broadcasting
    diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    return distance_matrix

def generate_file(num_cities, filename="tsp_instance.txt", x_range=(0, 2000), y_range=(0, 2000)):
    """
    Generates a TSP instance file with the specified number of cities.
    
    Parameters:
    - num_cities (int): The number of cities to generate.
    - filename (str): The output filename.
    - x_range (tuple): The range of x-coordinates (min, max).
    - y_range (tuple): The range of y-coordinates (min, max).
    """
    # Generate random coordinates for each city within the specified range
    x_coords = np.random.randint(x_range[0], x_range[1], num_cities)
    y_coords = np.random.randint(y_range[0], y_range[1], num_cities)

    # Write the coordinates to a file in the specified format
    with open(filename, 'w') as f:
        f.write(f"{num_cities}\n")  # First line is the number of cities
        for i in range(num_cities):
            f.write(f"{i + 1} {x_coords[i]} {y_coords[i]}\n")  # Write index, x, y

    print(f"Generated TSP file with {num_cities} cities: {filename}")

# Example usage:
#generate_file(5, filename="tsp_instance_5.txt")

