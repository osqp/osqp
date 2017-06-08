from __future__ import print_function

# For plotting
import matplotlib.colors as mc
from scipy.interpolate import griddata
import utils.plotting as plotting
import matplotlib.pyplot as plt


import numpy as np
from scipy.spatial import ConvexHull  # Create convex hull of PWL approximations

# Dataframes
import pandas as pd
from tqdm import tqdm

# CVXPY
import cvxpy


def get_performance_and_ratio(df):
    """
    Compute
        1) sample performance using their number of iterations related
           to the min one
        2) ratio tr(P)/tr(A'A) for the dataframe
    """

    df.loc[:, 'p'] = (df['iter'] - df['iter'].min()) / \
        (df['iter'].max() - df['iter'].min()) * 100

    df.loc[:, 'trPovertrAtA'] = df['trP'] / (df['froA'] * df['froA'])

    return df


def save_plot(df, name):
    """
    Plot behavior of 'name' in selected dataframe
    """

    # Dummy value always true
    location = (df['alpha'] > 0)

    # Get best iteration values (there are many) and
    # pick first pair sigma and alpha
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


def compute_pwl_lower_approx(x, y):
    """
    Compute pwl lower approximation of the convex function y = f(x)
    passing by the specified points
    """
    points = np.vstack((x, y)).T
    hull = ConvexHull(points)
    hull_eq = hull.equations
    A_hull = hull_eq[:, :2]
    b_hull = hull_eq[:, -1]

    # Delete line passing by first and last point
    i_idx = []
    for i in range(A_hull.shape[0]):
        if abs(A_hull[i, 0] * x[0] + A_hull[i, 1] * y[0] + b_hull[i]) < 1e-04:   # First point
            if abs(A_hull[i, 0] * x[-1] + A_hull[i, 1] * y[-1] + b_hull[i]) < 1e-04:   # Last point
                i_idx += [i]

    # Delete rows
    A_hull = np.delete(A_hull, i_idx, 0)
    b_hull = np.delete(b_hull, i_idx)

    # Return hyperplanes in form y >= A*x  + b
    A = np.divide(-A_hull[:, 0], A_hull[:, 1])
    b = np.divide(-b_hull[:], A_hull[:, 1])

    # Return convex hull in the form A_hull * x <= b_hull
    return A, b

'''
Main Script
'''

# Read results (only the ones less then max_iter)
# lasso = pd.read_csv('results/lasso_full.csv')
# nonneg_l2 = pd.read_csv('results/nonneg_l2_full.csv')
# portfolio = pd.read_csv('results/portfolio_full.csv')
# svm = pd.read_csv('results/svm_full.csv')
# res = pd.concat([lasso, portfolio, nonneg_l2, svm],
#                 ignore_index=True)

# Read full results
res = pd.read_csv('results/results_full.csv')

# Select problems not saturated at max number of iterations
res = res.loc[(res['iter'] < 1000)]

# Assign group headers
group_headers = ['seed', 'name']

print("Compute performance index and ratio for all problems")
tqdm.pandas()
# Assign performance and ratio to samples
problems = res.groupby(group_headers)
res_p = problems.progress_apply(get_performance_and_ratio)
problems_p = res_p.groupby(group_headers)

print("\nTotal number of problems: %i" % len(problems_p.groups))

'''
Build piecewise-linear (PWL) functions f_i(rho)
'''
# Create list of arrays
A = []
b = []

print("\nComputing PWL lower approximations")
for _, group in tqdm(problems_p):
    f = group['p'].values
    rho = group['rho'].values

    A_temp, b_temp = compute_pwl_lower_approx(rho, f)

    # Append arrays just found with list
    A.append(A_temp)
    b.append(b_temp)

# # DEBUG
# i = i - 1
#
# DEBUG: Try to test PWL functions of last group
# plt.figure()
# ax = plt.gca()
# rho_vec = np.linspace(0, 10, 100)
# plt.plot(rho, f)
# for j in range(len(b_temp)):
#     f_temp = A_temp[j] * rho_vec + b_temp[j]
#     plt.plot(rho_vec, f_temp)
# ax.set_xlim(0, 0.1)
# ax.set_ylim(0, 50)
# plt.show(block=False)

# import ipdb; ipdb.set_trace()

'''
Solve LP with CVXPY
'''
print("\n\nSolving problem with CVXPY and GUROBI")

# Solve for only n_prob problems
n_prob = len(problems_p.groups)

t = cvxpy.Variable(n_prob)
rho = cvxpy.Variable(n_prob)

# Line with offset
# x = cvxpy.Variable(2)

# Only line
x = cvxpy.Variable()

# Add linear cost
objective = cvxpy.Minimize(cvxpy.sum_entries(t))

# Add constraints
constraints = []

# Add inequality constraints
i = 0
print("Adding inequality constraints")
for _, problem in tqdm(problems_p):
    # Solve for only 10 problems
    if i < n_prob:
        for j in range(len(b[i])):
            constraints += [A[i][j] * rho[i] + b[i][j] <= t[i]]
    i += 1

# Add equality constraints
i = 0
print("Adding equality constraints")
for _, problem in tqdm(problems_p):
    if i < n_prob:
        ratio = problem['trPovertrAtA'].iloc[0]
        constraints += [x * ratio == rho[i]]

        # Line with offset
        # constraints += [x[0] + x[1] * ratio == rho[i]]
    i += 1

# Add constraints on rho
constraints += [rho >= 0]

# Define problem
prob = cvxpy.Problem(objective, constraints)

# Solve problem
prob.solve(solver=cvxpy.MOSEK, verbose=True)

'''
Create contour plot from 3D scatter plot with rho, ratio, efficiency
'''

# Get grid data
xi, yi, zi = get_grid_data(res_p['trPovertrAtA'], res_p['rho'], res_p['p'])

# Plot contour lines
# levels = [0., 0.25, 0.5, 0.75, 0.9, 0.95, 1.]
# levels = [0., 0.3, 0.6, 1., 5., 10., 20., 100.]
# levels = [0., 0.5, 1., 2., 3., 5., 10., 100.]
levels = [0., 3., 6., 10., 15., 20., 100.]


# use here 256 instead of len(levels)-1 becuase
# as it's mentioned in the documentation for the
# colormaps, the default colormaps use 256 colors in their
# definition: print(plt.cm.jet.N) for example
norm = mc.BoundaryNorm(levels, 256)

ax = plotting.create_figure(0.9)
plt.contour(xi, yi, zi, levels=levels, norm=norm, cmap=plt.cm.viridis_r)
plt.contourf(xi, yi, zi, levels=levels, norm=norm, cmap=plt.cm.viridis_r)
ax.set_ylabel(r'$\rho$')
ax.set_xlabel(r'$\frac{\mathrm{tr}(P)}{\mathrm{tr}(A^{T}A)}$')
ax.set_title(r'Performance $p$')
ax.set_xlim(0., 2.)
plt.colorbar()
plt.tight_layout()

'''
Plot fit line on the graph
'''
x_fit = np.asarray(x.value).flatten()
ratio_vec = np.linspace(0, 2., 100)
rho_fit_vec = x_fit * ratio_vec
# rho_fit_vec = x_fit[0] + x_fit[1] * ratio_vec
plt.plot(ratio_vec, rho_fit_vec, color='C3')

plt.show(block=False)
plt.savefig('behavior.pdf')
