import pandas
import numpy as np

# Load the data from problem 3 in homework 2
df_training_data = pandas.read_csv('data.csv')

global v_a
global v_b
global v_1

v_a = 0.3597
v_b = -0.8111
v_1 = 0.5619

my_model = (v_a, v_b, v_1)

def calculate_ln_r(model: tuple) -> tuple:
    # Load the training data
    df = pandas.read_csv('data.csv')

    # calculate v * xi
    df['v_xi'] = df[['ai']] * model[0]

    # calculate p_i(v), and the inverse q_i(v)
    df['pi'] = 1 / (1 + np.exp(-df[['v_xi']]))
    df['qi'] = 1 - df['pi']

    # assign the value of column 'ri' based on whether yi is True or False
    df.loc[df['yi'] == True, 'ri'] = df['pi']
    df.loc[df['yi'] == False, 'ri'] = df['qi']

    r = np.sum(df['ri'])
    ln_r = np.log(r)

    return df, r, ln_r


test = calculate_ln_r(my_model)