import pandas as pd

df = pd.read_csv('b52.csv', names=['ant_number', 'iterations', 'pheromone_evaporation', 'alpha', 'beta', 'path' ,'result'])

lowest_rows = df.nsmallest(5, 'result')
result = lowest_rows.iloc[:, list(range(5)) + [-1]]
result.to_csv('b52r.csv', index=False)