import pandas as pd

df = pd.read_csv('data/bier/bier127.csv', names=['ant_number', 'iterations', 'pheromone_evaporation', 'alpha', 'beta', 'path' ,'result'])

lowest_rows = df.nsmallest(5, 'result')
result = lowest_rows.iloc[:, list(range(5)) + [-1]]
result.to_csv('data/bier/result-v1.csv', index=False)