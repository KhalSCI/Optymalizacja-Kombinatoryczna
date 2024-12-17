from ant import ACO
import pandas as pd
import random
import pandas as pd
from multiprocessing import Pool


def run_aco(ant_number, iterations, pheromone_evaporation, alpha_one, alpha_two, beta, epsilon):
    aco = ACO(ant_number=ant_number, iterations=iterations, pheromone_evaporation=pheromone_evaporation, alpha_one=alpha_one, alpha_two=alpha_two, beta=beta, epsilon=epsilon)
    aco.init_matrix('data/berlin/berlin52.txt', 'data/berlin/berlin52_cost_matrix.csv', pheromone_one_start=0.200,
                    pheromone_two_start=0.200, visibility_const=200)
    return aco.run()

def analyze_results(df):
    if df.empty:
        return {
            'ant_number': 65,
            'iterations': 100,
            'pheromone_evaporation': 0.4,
            'alpha_one': 1.5,
            'alpha_two': 2.5,
            'beta': 3.5,
            'epsilon': 0.05
        }

    best_result = df['result'].min()
    best_params = df[df['result'] == best_result].iloc[0]

    return {
        'ant_number': int(best_params['ant_number']),
        'iterations': int(best_params['iterations']),
        'pheromone_evaporation': float(best_params['pheromone_evaporation']),
        'alpha_one': float(best_params['alpha_one']),
        'alpha_two': float(best_params['alpha_two']),
        'beta': float(best_params['beta']),
        'epsilon': float(best_params['epsilon'])
    }


def adjust_parameters(best_params):
    ant_number = best_params['ant_number'] + random.randint(-10, 20)
    iterations = best_params['iterations'] + random.randint(-10, 20)
    pheromone_evaporation = best_params['pheromone_evaporation'] + random.uniform(-0.1, 0.3)
    alpha_one = best_params['alpha_one'] + random.uniform(-1.0, 1.0)
    alpha_two = best_params['alpha_two'] + random.uniform(-1.0, 1.0)
    beta = best_params['beta'] + random.uniform(-1.0, 1.0)
    epsilon = best_params['epsilon'] + random.uniform(-0.05, 0.05)

    ant_number = max(10, min(100, ant_number))
    iterations = max(10, min(200, iterations))
    pheromone_evaporation = max(0.1, min(0.9, pheromone_evaporation))
    alpha_one = max(0.1, min(5.0, alpha_one))
    alpha_two = max(0.1, min(5.0, alpha_two))
    beta = max(1.0, min(10.0, beta))
    epsilon = max(0.01, min(0.1, epsilon))

    return ant_number, iterations, pheromone_evaporation, alpha_one, alpha_two, beta, epsilon


def run_aco_helper(params):
    ant_number, iterations, pheromone_evaporation, alpha_one, alpha_two, beta, epsilon = params
    aco = ACO(ant_number, iterations, pheromone_evaporation, alpha_one, alpha_two, beta, epsilon)
    aco.init_matrix('data/berlin/berlin52.txt', 'data/berlin/berlin52_cost_matrix.csv', 0.2, 0.2, 200)
    result = aco.run()
    return result


def run_multiple():
    try:
        df = pd.read_csv('data/berlin/berlin52-test.csv',
                         names=['ant_number', 'iterations', 'pheromone_evaporation', 'alpha_one', 'alpha_two', 'beta',
                                'epsilon', 'path', 'result'])
        best_params = analyze_results(df)
    except FileNotFoundError:
        best_params = {
            'ant_number': 65,
            'iterations': 100,
            'pheromone_evaporation': 0.4,
            'alpha_one': 1.5,
            'alpha_two': 2.5,
            'beta': 3.5,
            'epsilon': 0.05
        }

    params_list = [adjust_parameters(best_params) for _ in range(10)]

    with Pool(processes=3) as pool:
        results = pool.map(run_aco_helper, params_list)

    df = pd.DataFrame(results)
    df.to_csv('data/berlin/berlin52-test.csv', mode='a', header=False, index=False)


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    run_multiple()