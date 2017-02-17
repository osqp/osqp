import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.width', 1000)

from scipy.optimize import curve_fit

# Define candidate function to be fit
def func_iter(x, c0, c1, c2, c3, c4):

    return c0*np.power(x[0], c1*x[3] + c2*x[4])*np.power(x[1], c3)*np.power(x[2], c4)

# Main function
if __name__ == '__main__':

    # Read results (only the ones less then max_iter)
    res = pd.read_csv('tests/qp_problems/results/full_results.csv')
    res = res.loc[(res['iter'] < 2400)]

    # Define inputs and output
    inputs = ['rho', 'sigma', 'alpha', 'n', 'm']
    output = ['iter']

    # Extract input vector to fit
    xdata = res[inputs].values.T
    ydata = res[output].values.flatten()

    # Fit function
    popt, pcov = curve_fit(func_iter, xdata, ydata,
                           bounds=(-20*np.ones(5),
                                   20*np.ones(5)))

    perr = np.sqrt(np.diag(pcov))
