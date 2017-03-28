from __future__ import print_function

# For plotting
import matplotlib as mpl
import matplotlib.colors as mc
# mpl.use('Agg')  # For plotting on remote server
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('axes', titlesize=18)   # fontsize of the tick labels
plt.rc('xtick', labelsize=15)   # fontsize of the tick labels
plt.rc('ytick', labelsize=15)   # fontsize of the tick labels
plt.rc('legend', fontsize=15)   # legend fontsize
plt.rc('text', usetex=True)     # use latex
plt.rc('font', family='serif')

# Numerics
import numpy as np

# Dataframes
import pandas as pd
from tqdm import tqdm

# CVXPY
import cvxpy


def get_performance(df):
    """
    Compute sample performance using their number of iterations related
    to the min one
    """
    # df.loc[:, 'p'] = df['iter'] / df['iter'].min()
    df.loc[:, 'p'] = (df['iter'] - df['iter'].min()) / \
        (df['iter'].max() - df['iter'].min()) * 100

    return df


def get_ratio(df):
    """
    Get ratio tr(P)/tr(A'A) for the dataframe
    """
    df.loc[:, 'trPovertrAtA'] = df['trP'] / (df['froA'] * df['froA'])
    return df


def save_plot(df, name):
    """
    Plot behavior of 'name' in selected dataframe
    """

    # Dummy value always true
    location = (df['alpha'] > 0)

    # Get best iteration values (there are many) and pick first pair sigma and alpha
    if name is not 'sigma':
        test_sigma = df.loc[(df['p'] == 1.)].sigma.values[-1]
        location &= (df['sigma'] == test_sigma)
    if name is not 'alpha':
        test_alpha = df.loc[(df['p'] == 1.)].alpha.values[-1]
        location &= (df['alpha'] == test_alpha)
    if name is not 'rho':
        test_rho = df.loc[(df['p'] == 1.)].rho.values[-1]
        location &= (df['rho'] == test_rho)

    # Get test case in specified location
    test_case = df.loc[location]

    # Plot behavior
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    ax = plt.gca()
    if name is 'rho':
        ax.set_xscale('log')
    plt.scatter(test_case[name], test_case['iter'])
    ax.set_ylabel('iter')
    ax.set_xlabel(name)
    plt.grid()
    plt.show(block=False)

    plt.subplot(1, 2, 2)
    ax = plt.gca()
    if name is 'rho':
        ax.set_xscale('log')
    plt.scatter(test_case[name], test_case['p'])
    ax.set_ylabel('weight')
    ax.set_xlabel(name)
    plt.grid()
    plt.show(block=False)

    plt.tight_layout()
    plt.savefig('figures/%s.pdf' % name)


def get_grid_data(x, y, z, resX=100, resY=100):
    "Convert 3 column data to matplotlib grid"
    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')
    # X, Y = np.meshgrid(xi, yi)
    return xi, yi, zi


# Main function
if __name__ == '__main__':

    # Read results (only the ones less then max_iter)
    # lasso = pd.read_csv('results/lasso_full.csv')
    # nonneg_l2 = pd.read_csv('results/nonneg_l2_full.csv')
    # portfolio = pd.read_csv('results/portfolio_full.csv')
    # svm = pd.read_csv('results/svm_full.csv')
    # res = pd.concat([lasso, portfolio, nonneg_l2, svm],
    #                 ignore_index=True)

    res = pd.read_csv('results/results_full.csv')

    # Select problems not saturated at max number of iterations
    # res = res.loc[(res['iter'] < 2499)]

    # Assign group headers
    group_headers = ['seed', 'name']

    # Assign efficienct to samples
    problems = res.groupby(group_headers)
    res_p = problems.apply(get_performance)
    problems_p = res_p.groupby(group_headers)

    '''
    Create contour plot from 3D scatter plot with rho, ratio, efficiency
    '''
    # Get ratio for each group
    res_p = problems_p.apply(get_ratio)

    # Get grid data
    xi, yi, zi = get_grid_data(res_p['trPovertrAtA'], res_p['rho'], res_p['p'])

    # Plot contour lines
    # levels = [0., 0.25, 0.5, 0.75, 0.9, 0.95, 1.]
    levels = [0., 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1., 100.]

    # use here 256 instead of len(levels)-1 becuase
    # as it's mentioned in the documentation for the
    # colormaps, the default colormaps use 256 colors in their
    # definition: print(plt.cm.jet.N) for example
    norm = mc.BoundaryNorm(levels, 256)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.contour(xi, yi, zi, levels=levels, norm=norm, cmap=plt.cm.jet_r)
    plt.contourf(xi, yi, zi, levels=levels, norm=norm, cmap=plt.cm.jet_r)
    ax.set_ylabel(r'$\rho$')
    ax.set_xlabel(r'$\frac{{\rm tr}(P)}{{\rm tr}(A^{T}A)}$')
    ax.set_title(r'Performance $p$')
    plt.colorbar()
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig('behavior.pdf')

    '''
    Build piecewise-linear (PWL) functions f_i(rho)

    n_prob is the number of problems generated (identified by a seed)
    n_rho is the number of different rho values
        N.B. We need to fit n_rho - 1 linear pieces.

    '''
    # get number of problems
    n_prob = len(problems_p.groups)

    # get number of rho elements per problem
    n_rho = problems_p.size().iloc[0]  # Number of elements in first problem

    a = np.zeros((1, n_prob, n_rho - 1))
    b = np.zeros((n_prob, n_rho - 1))

    i = 0
    for name, group in problems_p:
        f = group['p'].values
        rho = group['rho'].values
        for j in range(n_rho - 1):
            # TODO: Adapt and check!
            a[0, i, j] = (f[j + 1] - f[j]) / (rho[j + 1] - rho[j])
            b[i, j] = f[j] - a[0, i, j] * rho[j]

        # Increase problem counter
        i += 1

    # # DEBUG
    # i = i - 1
    #
    # # DEBUG: Try to test PWL functions of last group
    # plt.figure()
    # ax = plt.gca()
    # rho_vec = np.linspace(0, 10, 100)
    # plt.plot(rho, f)
    # for j in range(n_rho - 1):
    #     if f[j] < 0.5:
    #         f_temp = a[0, i, j] * rho_vec + b[i, j]
    #         plt.plot(rho_vec, f_temp)
    # ax.set_xlim(0, 0.02)
    # ax.set_ylim(0, 6)
    # plt.show(block=False)


    '''
    Solve LP with CVXPY
    '''
    t = cvxpy.Variable(n_prob)
    rho = cvxpy.Variable(n_rho)
    x = cvxpy.Variable(2)

    # Add linear cost
    cost = cvxpy.Maximize(cvxpy.sum_entries(t))

    # Add equality constraints
    # constraints = []
    # for i in range(n_prob):
    #     constraints
