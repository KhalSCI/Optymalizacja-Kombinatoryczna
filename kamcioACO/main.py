import pandas as pd

df = pd.read_csv('./data/berlin/berlin52-test.csv', names=['ant_number', 'iterations', 'pheromone_evaporation', 'alpha_one', 'alpha_two', 'beta', 'epsilon', 'path' ,'result'])

lowest_rows = df.nsmallest(5, 'result')
result = lowest_rows.iloc[:, list(range(7)) + [-1]]
result.to_csv('test-berlin.csv', index=False)