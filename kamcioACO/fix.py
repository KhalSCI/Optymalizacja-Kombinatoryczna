import re

# Path to the file
file_path = 'data/benchmark_instances_transformed/transformed_d1291.tsp'

# Read the file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Modify all lines
for i, line in enumerate(lines):
    parts = line.split()
    if len(parts) == 3:
        parts[1] = str(int(float(parts[1])))
        parts[2] = str(int(float(parts[2])))
        lines[i] = ' '.join(parts) + '\n'

# Write the changes back to the file
with open(file_path, 'w') as file:
    file.writelines(lines)