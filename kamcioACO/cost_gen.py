import numpy as np

def generate_cost_matrix(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
        num_cities = int(lines[0])
        coordinates = []
        for line in lines[1:num_cities + 1]:
            parts = line.strip().split()
            coordinates.append((int(parts[1]), int(parts[2])))

    dist_matrix = np.linalg.norm(
        np.array(coordinates)[:, None, :] - np.array(coordinates)[None, :, :], axis=-1
    )
    np.fill_diagonal(dist_matrix, np.inf)  # Cannot travel to itself

    cost_matrix = dist_matrix * np.random.uniform(1, 2, size=dist_matrix.shape)
    np.savetxt(output_file, cost_matrix, delimiter=',')

# Example usage
generate_cost_matrix('data/other/tsp1000.txt', 'data/other/tsp1000_cost_matrix.csv')