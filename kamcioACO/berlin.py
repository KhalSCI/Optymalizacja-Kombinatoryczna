from ant import ACO
import pandas as pd
import random


def run_aco(ant_number, iterations, pheromone_evaporation, alpha_one, alpha_two, beta, epsilon):
    aco = ACO(ant_number=ant_number, iterations=iterations, pheromone_evaporation=pheromone_evaporation, alpha_one=alpha_one, alpha_two=alpha_two, beta=beta, epsilon=epsilon)
    aco.init_matrix("data/berlin/berlin52.txt", pheromone_one_start=0.30000, pheromone_two_start=0.5000, visibility_const=200)
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
    ant_number = best_params['ant_number'] + random.randint(-3, 7)
    iterations = best_params['iterations'] + random.randint(-3, 7)
    pheromone_evaporation = best_params['pheromone_evaporation'] + random.uniform(-0.06000000, 0.06000000)
    alpha_one = best_params['alpha_one'] + random.uniform(-0.0001000000, 0.05000000000)
    alpha_two = best_params['alpha_two'] + random.uniform(-0.000010000000, 0.03000000000)
    beta = best_params['beta'] + random.uniform(-0.020000000000, 0.0000100000)
    epsilon = best_params['epsilon'] + random.uniform(-0.01, 0.01)

    ant_number = max(52, ant_number)
    iterations = max(40, iterations)
    pheromone_evaporation = min(max(0.1, pheromone_evaporation), 0.9)
    alpha_one = min(max(0.3, alpha_one), 4.5)
    alpha_two = min(max(0.3, alpha_two), 4.5)
    beta = min(max(2.0, beta), 6.0)
    epsilon = min(max(0.01, epsilon), 0.3)

    return ant_number, iterations, pheromone_evaporation, alpha_one, alpha_two, beta, epsilon


from multiprocessing import Pool

def run_aco_helper(params):
    ant_number, iterations, pheromone_evaporation, alpha_one,alpha_two, beta, epsilon = params
    return run_aco(ant_number, iterations, pheromone_evaporation, alpha_one, alpha_two, beta, epsilon)

def run_multiple():
    results = []

    try:
        df = pd.read_csv('/data/berlin/b52-test.csv',
                         names=['ant_number', 'iterations', 'pheromone_evaporation', 'alpha_one', 'alpha_two', 'beta', 'epsilon', 'path', 'result'])
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

    params_list = [adjust_parameters(best_params) for _ in range(30)]

    with Pool(processes=3) as pool:
        results = pool.map(run_aco_helper, params_list)

    df = pd.DataFrame(results)
    df.to_csv('data/berlin/berlin52-test.csv', mode='a', header=False, index=False)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    run_multiple()