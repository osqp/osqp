from __future__ import print_function
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
pd.set_option('display.width', 1000)


# import sklearn tools
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline   # Make pipeline for estimators
from sklearn.preprocessing import PolynomialFeatures  # Construct polynomials
from sklearn.linear_model import (LinearRegression, HuberRegressor, Ridge)
from sklearn.metrics import mean_squared_error

# # Define candidate function to be fit
# def func_iter(x, c0, c1, c2, c3, c4):
#
#     return c0*np.power(x[0], c1*x[3] + c2*x[4])*np.power(x[1], c3)*np.power(x[2], c4)

# Main function
if __name__ == '__main__':

    # Read results (only the ones less then max_iter)
    res = pd.read_csv('tests/qp_problems/results/full_results.csv')
    res = res.loc[(res['iter'] < 2400)]

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


    resz = res.loc[(res['m'] == 60) &
                   (res['n'] == 60) &
                   (abs(res['rho'] - 0.006952) < 1e-04 ) &
                   (abs(res['sigma'] - 14.384499) < 1e-04) &
                   (abs(res['alpha'] - 1.521053) < 1e-04)]

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
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.tripcolor(resz.froA, resz.froP, resz.iter)
    ax.set_xlabel('froA')
    ax.set_ylabel('froP')
    ax.set_title('Iterations vs froA and froP with fixed rho, sigma, alpha')
    plt.colorbar()
    plt.show(block=False)


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
