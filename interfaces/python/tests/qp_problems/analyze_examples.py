import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
from ipdb import set_trace
pd.set_option('display.width', 1000)



class MinimumItersParameters(object):
    """
    Class containing arrays of minimum parameters for each value of a field
    """
    def __init__(self, iter_field_min, rho_field_min,
                 sigma_field_min, alpha_field_min, field_vals,
                 field_name):
        self.iter_field_min = iter_field_min
        self.rho_field_min = rho_field_min
        self.sigma_field_min = sigma_field_min
        self.alpha_field_min = alpha_field_min
        self.field_vals = field_vals
        self.field_name = field_name



# Get best rho and iters given results dataframe
def get_best_iters(results, field):
    """
    Get best iterations, rho and alpha values for specified field over all simulations
    """
    # Get data vectors from results
    field_vals = np.unique(results[field].values)  # values of fields vec

    # Rho and iter arrays
    alpha_field_min = []
    rho_field_min = []
    sigma_field_min = []
    iter_field_min = []

    # Index of elements to remove from field_vals if the maximum
    # number of iterations is hit
    idx_to_remove = []

    # iterate over the values of the field
    for idx in range(len(field_vals)):  # Iterate over possible n values
        # Get results for the required field value
        results_field = results.loc[(results[field] == field_vals[idx])]
        iter_field = results_field['iter'].values
        rho_field = results_field['rho'].values
        sigma_field = results_field['sigma'].values
        alpha_field = results_field['alpha'].values

        # Get index of minimum
        # N.B. Use it only if the minimum number of iterations is less
        #      than tha maximum ones
        min_field_idx = iter_field.argmin()
        if iter_field[min_field_idx] < 2400:
            # print('min_iter = %d' % iter_field[min_field_idx], end='\n')
            iter_field_min.append(iter_field.min())
            rho_field_min.append(rho_field[min_field_idx])
            sigma_field_min.append(sigma_field[min_field_idx])
            alpha_field_min.append(alpha_field[min_field_idx])
        else:
            idx_to_remove.append(idx)

    # Remove values for which we hit the maximum number of iterations
    field_vals = np.delete(field_vals, idx_to_remove)

    # Transform to arrays
    iter_field_min = np.array(iter_field_min)
    rho_field_min = np.array(rho_field_min)
    sigma_field_min = np.array(sigma_field_min)
    alpha_field_min = np.array(alpha_field_min)

    return MinimumItersParameters(iter_field_min, rho_field_min,
                                  sigma_field_min, alpha_field_min, field_vals, field)

def plot_best_iter(params):
    
    # Plot
    plt.figure()
    ax = plt.gca()
    ax.plot(params.field_vals, params.rho_field_min, 'o')
    ax.set_yscale('log')
    plt.xlabel('Data %s' % params.field_name)
    plt.ylabel(r'Best $\rho$')
    plt.title(r"Best $\rho$ against %s" % params.field_name)
    plt.grid()
    plt.show(block=False)

    # Plot
    plt.figure()
    ax = plt.gca()
    ax.plot(params.field_vals, params.sigma_field_min, 'o')
    ax.set_yscale('log')
    plt.xlabel('Data %s' % params.field_name)
    plt.ylabel(r'Best $\sigma$')
    plt.title(r"Best $\sigma$ against %s" % params.field_name)
    plt.grid()
    plt.show(block=False)

    # Plot
    plt.figure()
    ax = plt.gca()
    ax.plot(params.field_vals, params.alpha_field_min, 'o')
    # ax.set_yscale('log')
    plt.xlabel('Data %s' % params.field_name)
    plt.ylabel(r'Best $\alpha$')
    plt.title(r"Best $\alpha$ against %s" % params.field_name)
    plt.grid()
    plt.show(block=False)




# Main function
if __name__ == '__main__':

    # Read results
    res = pd.read_csv('tests/qp_problems/results/full_results.csv')

    # Choose data_field to plot
    data_field = 'm'

    # Get best iterations for certain parameter
    min_iter_params = get_best_iters(res, data_field)

    # Plot parameters
    plot_best_iter(min_iter_params)
