import pandas
import numpy as np
# This is so that numpy will be quiet
np.seterr(divide='ignore')
import tqdm

v_a = 0.3597
v_b = -0.8111
v_1 = 0.5619

my_model = (v_a, v_b, v_1)

def measure_model(model: tuple) -> tuple:
    """
    This function measures the performance of a given model
    """
    # Load the training data
    df = pandas.read_csv('data.csv')

    # calculate v dot xi
    df['v_xi'] = np.dot(df[['ai', 'bi', '1']],model)
    df['v_xi'].astype(float)

    # calculate p_i(v), and the inverse q_i(v)
    df['pi'] = 1 / (1 + np.exp(-1 * df[['v_xi']]))
    df['pi'].astype(float)
    df['qi'] = 1 - df['pi']
    df['qi'].astype(float)

    # assign the value of column 'ri' based on whether yi is True or False
    df.loc[df['yi'] == True, 'ri'] = df['pi']
    df.loc[df['yi'] == False, 'ri'] = df['qi']
    df['ri'].astype(float)

    # Calculate r and ln_r as a measure of the model's performance.
    r = np.product(df['ri'])
    ln_r = np.log(r)

    return df, r, ln_r

def randomly_find_model(n: int, max: float = 10, min: float = -10) -> tuple:
  """
  This function naively finds a model that fits the data by making random
  guesses and keeping the one that performs best
  """

  random_models = np.random.uniform(low=min, high=max, size=(n, 3))

  best_model = (0, 0, 0)
  best_score = -np.inf
  for model in tqdm.tqdm(random_models, desc="Randomly trying to find model", total=n):     
    model_performance = measure_model(model)
    if best_score < model_performance[2]:
       best_score = model_performance[2]
       best_model = model

  return best_model, best_score

model = randomly_find_model(10000)

print(f"The best model found was {model[0]} with a score of {model[1]}")