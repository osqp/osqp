from __future__ import print_function

# For plotting
import matplotlib as mpl
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

# # import sklearn tools
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import make_pipeline   # Make pipeline for estimators
# from sklearn.preprocessing import PolynomialFeatures  # Construct polynomials
# from sklearn.linear_model import (LinearRegression, HuberRegressor, Ridge)
# from sklearn.metrics import mean_squared_error
#
# # # Define candidate function to be fit
# # def func_iter(x, c0, c1, c2, c3, c4):
# #
# #     return c0*np.power(x[0], c1*x[3] + c2*x[4])* \
# #                        np.power(x[1], c3)*np.power(x[2], c4)


# def get_rho_and_tr_ratio(df):
#     """
#     Get best rho and ratio of traces tr(P)/tr(A'A)
#     """
#     # Get best parameters
#     df_best = get_best_params(df)
#
#
#     # get values
#     trP = df_best['trP']
#     froA = df_best['froA']
#     trAtA = froA.values * froA.values
#
#
#     # Compute ratio
#     trP_over_trAtA = trP / trAtA
#
#     # IT IS HAPPENING HERE!!!!
#     # import ipdb; ipdb.set_trace()
#
#     # Get return dataframe
#     df_return = df_best.loc[:, ['seed', 'n', 'm', 'rho', 'trP', 'froA']]
#     df_return.loc[:, 'trAtA'] = pd.Series(trAtA,
#                                           index=df_return.index)
#     df_return.loc[:, 'n_over_m'] = pd.Series(df_best['n']/df_best['m'],
#                                                        index=df_return.index)
#     df_return.loc[:, 'trP_over_trAtA'] = pd.Series(trP_over_trAtA,
#                                                    index=df_return.index)
#
#
#     # df_return['trAtA'] = trAtA
#     # df_return['trP_over_trAtA'] = trP_over_trAtA
#
#     # # Return new dataframe
#     # df_return = pd.DataFrame({'seed': [df_best['seed']],
#     #                           'trP': [df_best['trP']],
#     #                           'froA': [df_best['froA']],
#     #                           'trAtA': [df_best['froA'] ** 2],
#     #                           'trP_over_trAtA': [trP_over_trAtA]})
#
#     # import ipdb; ipdb.set_trace()
#
#     # return new dataframe
#     return df_return


# def get_best_params(df):
#     """
#     Transform weighted frame into another frame with best parameters
#     """
#     # Get best parameters
#     df_best = df.loc[df['e'] == 1.]
#
#     # Get highest sigma
#     max_sigma = df_best['sigma'].max()
#
#     # Get best row
#     df_best = df_best.loc[(df_best['sigma'] == max_sigma)]
#
#     if len(df_best) > 1:  # If multiple values choose one with min alpha
#         min_alpha = df_best['alpha'].min()
#         df_best = df_best.loc[(df_best['alpha'] == min_alpha)]
#
#     if len(df_best) > 1:  # If multiple values chose one with min rho
#         # Choose min rho element
#         rho_best = df_best['rho'].min()
#
#         if rho_best > 5:
#             import ipdb; ipdb.set_trace()
#         # rho_closest_to_min = np.abs(df_best['rho'] - min_rho).min()
#         df_best = df_best.loc[(df_best['rho'] == rho_best)]
#         # import ipdb; ipdb.set_trace()
#         # df_best = df_best.iloc[[idx]]
#
#
#     # import ipdb; ipdb.set_trace()
#
#     return df_best


def get_efficiency(df):
    """
    Compute sample efficiency using their number of iterations related
    to the min one
    """
    df.loc[:, 'e'] = df['iter'].min() / df['iter']
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
        test_sigma = df.loc[(df['e'] == 1.)].sigma.values[-1]
        location &= (df['sigma'] == test_sigma)
    if name is not 'alpha':
        test_alpha = df.loc[(df['e'] == 1.)].alpha.values[-1]
        location &= (df['alpha'] == test_alpha)
    if name is not 'rho':
        test_rho = df.loc[(df['e'] == 1.)].rho.values[-1]
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
    plt.scatter(test_case[name], test_case['e'])
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
    lasso = pd.read_csv('results/lasso_full.csv')
    nonneg_l2 = pd.read_csv('results/nonneg_l2_full.csv')
    portfolio = pd.read_csv('results/portfolio_full.csv')
    svm = pd.read_csv('results/svm_full.csv')
    # res = pd.concat([lasso, nonneg_l2, portfolio, svm])
    res = pd.concat([lasso, portfolio, nonneg_l2, svm],
                    ignore_index=True)

    # res = pd.read_csv('results/results_full.csv')

    # Select problems not saturated at max number of iterations
    res = res.loc[(res['iter'] < 2499)]

    # Problem headings
    # headings = ['n', 'm', 'name', 'seed']

    # Group problems
    # problems = res.groupby(headings)
    # n_problems = len(problems.groups)

    # Assign efficienct to samples
    problems = res.groupby(['seed'])
    res_e = problems.apply(get_efficiency)
    problems_e = res_e.groupby(['seed'])

    # '''
    # Plot behavior for fixed sigma and alpha and changing rho
    # '''
    # test_name = (50.0, 60.0, 'svm', 3076953921.0)
    # test_name = (50.0, 60.0, 'svm', 107769053.0)
    # test_name = (40.0, 40.0, 'lasso', 685148778.0)
    # test_name = (94.0, 96.0, 'svm', 792958971.0)
    # test_name = (40.0, 40.0, 'lasso', 4089288235.0)

    # test_instance = problems_w.get_group(test_name)

    # Save plots for rho, sigma and alpha
    # save_plot(test_instance, 'rho')
    # save_plot(test_instance, 'sigma')
    # save_plot(test_instance, 'alpha')

    # Try to estimate value
    # ex_val = test_instance.iloc[0]
    # trP = ex_val['trP']
    # froA = ex_val['froA']
    # froA2 = froA * froA


    # '''
    # Plot behavior of optimal rho compared to tr(P)/tr(A'A)
    # '''

    # Compute best rho and ratio tr(P)/tr(A'A) for each group
    # rho_and_tr_ratio = pd.DataFrame()
    # for name, group in tqdm(problems_w):
    #     rho_and_tr_ratio = \
    #         rho_and_tr_ratio.append(get_rho_and_tr_ratio(group))


    # Plot
    # plt.figure()
    # ax = plt.gca()
    # plt.scatter(rho_and_tr_ratio['trP_over_trAtA'],
    #             rho_and_tr_ratio['rho'])
    # # ax.set_yscale('log')
    # ax.set_ylabel('rho')
    # ax.set_xlabel('trP_over_trAtA')
    # plt.grid()
    # plt.show(block=False)
    # plt.savefig('behavior.pdf')

    '''
    Create contour plot from 3D scatter plot with rho, ratio, efficiency
    '''
    # Get ratio for each group
    res_e = problems_e.apply(get_ratio)

    # Get grid data
    xi, yi, zi = get_grid_data(res_e['trPovertrAtA'], res_e['rho'], res_e['e'])

    # Plot contour lines
    levels = [0., 0.25, 0.5, 0.75, 0.9, 0.95, 1.]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.contour(xi, yi, zi, levels)
    plt.contourf(xi, yi, zi, levels)
    ax.set_ylabel(r'$\rho$')
    ax.set_xlabel(r'$\frac{{\rm tr}(P)}{{\rm tr}(A^{T}A)}$')
    ax.set_title(r'Efficiency $e$')
    plt.colorbar()
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig('behavior.pdf')

    # Get grid data
    # X, Y, Z = get_grid_data(res_e['rho'], res_e['trPovertrAtA'], res_e['e'])
    # plt.contourf(X, Y, Z)



    # Plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(res_e['rho'], res_e['trPovertrAtA'], res_e['e'])
    # plt.show(block=False)

    # # Plot
    # # x = np.linspace(0.0002, 0.05)
    # # y = 16 * x
    # plt.figure()
    # ax = plt.gca()
    # plt.scatter(rho_and_tr_ratio['n_over_m'],
    #             rho_and_tr_ratio['rho'])
    # # plt.plot(x, y)
    # # ax.set_yscale('log')
    # # ax.set_xscale('log')
    # ax.set_ylabel('rho')
    # ax.set_xlabel('n_over_m')
    # # ax.set_xlim(-.3, .3)
    # plt.grid()
    # plt.show(block=False)


    # Get optimal parameters for lasso problem
    # same_type_probs = res_w.groupby(['name'])
    # lasso_probs = same_type_probs.get_group(('lasso'))
    # best_lasso = lasso_probs.groupby(['seed']).apply(get_best_params)
    # pd.tools.plotting.scatter_matrix(best_lasso)



    # Select smaller dataset and consider only n, m, trP
    # res = res.loc[(res['m'] < 100) & (res['n'] < 100)]



    # Get only some features
    # features = ['n', 'm', 'rho', 'sigma', 'alpha', 'iter']
    # res = res[features]


    # Try to get one value
    # resz = res.loc[(res['m'] == 20) &
    #                (res['n'] == 10) &
    #                (abs(res['rho'] - 0.206913808111) < 1e-04 ) &
    #                (abs(res['sigma'] - 0.016238) < 1e-04) &
    #                (abs(res['alpha'] - 0.668421) < 1e-04)]


    #  resz = res.loc[(res['m'] == 60) &
                   #  (res['n'] == 60) &
                   #  (abs(res['rho'] - 0.006952) < 1e-04 ) &
                   #  (abs(res['sigma'] - 14.384499) < 1e-04) &
                   #  (abs(res['alpha'] - 1.521053) < 1e-04)]

    # resz = res.loc[(res['m'] == 30) &
    #                (res['n'] == 30) &
    #                (abs(res['rho'] - 0.006952) < 1e-04 ) &
    #                (abs(res['sigma'] - 14.384499) < 1e-04) &
    #                (abs(res['alpha'] - 1.521053) < 1e-04)]

    # resz[['trP', 'iter']].plot(x='trP', y='iter', style='o')
    # plt.show(block=False)
    #
    # resz[['froP', 'iter']].plot(x='froP', y='iter', style='o')
    # plt.show(block=False)
    #
    # resz[['trA', 'iter']].plot(x='trA', y='iter', style='o')
    # plt.show(block=False)
    # resz[['froA', 'iter']].plot(x='froA', y='iter', style='o')
    # plt.show(block=False)


    # Plot contour in 2D
    #
    #  fig = plt.figure()
    #  ax = fig.add_subplot(111)
    #  plt.tripcolor(resz.froA, resz.froP, resz.iter)
    #  ax.set_xlabel('froA')
    #  ax.set_ylabel('froP')
    #  ax.set_title('Iterations vs froA and froP with fixed rho, sigma, alpha')
    #  plt.colorbar()
    #  plt.show(block=False)


    # Plot 3D
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_trisurf(resz.froA, resz.froP, resz.iter)
    # ax.set_xlabel('froA')
    # ax.set_ylabel('froP')
    # ax.set_zlabel('iter')
    # plt.show(block=False)




    # # Fit curve (SKLEARN)
    # # Split dataset in train and test (randomly)
    # train, test = train_test_split(res, test_size = 0.2)
    # X = train[features[:-1]].values
    # y = train['iter'].values
    # X_test = test[features[:-1]].values
    # y_test = test['iter'].values
    #
    # estimators = [('Ridge', HuberRegressor()),
    #             #   ('Huber', HuberRegressor()),
    #               ('Linear', LinearRegression())
    #               ]
    #
    #
    # for name, estimator in estimators:
    #     model = make_pipeline(PolynomialFeatures(3), estimator)
    #     model.fit(X, y)
    #     mse = mean_squared_error(model.predict(X_test), y_test)
    #     print("%s mse = %.4e" % (name, mse))


    # Fit curve  (SCIPY curve_fit)
    # # Define inputs and output
    # inputs = ['rho', 'sigma', 'alpha', 'n', 'm']
    # output = ['iter']
    # # Extract input vector to fit
    # xdata = res[inputs].values.T
    # ydata = res[output].values.flatten()
    #
    # # Fit function
    # popt, pcov = curve_fit(func_iter, xdata, ydata,
    #                        bounds=(-20*np.ones(5),
    #                                20*np.ones(5)))
    #
    # perr = np.sqrt(np.diag(pcov))
