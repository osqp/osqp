# For plotting
import matplotlib.colors as mc
from scipy.interpolate import griddata
import utils.plotting as plotting
import matplotlib.pyplot as plt

import numpy as np
import scipy.sparse as spa

# Dataframes
import pandas as pd
from tqdm import tqdm

# Formulate problem
import cvxpy


def get_ratio_and_bounds(df):
    """
    Compute (relative to the current group)
        1) scaled number of iterations in [1, 100]
        2) ratio tr(P)/tr(A'A)
        3) lower and upper bounds for rho (between 0 and 10)
    """

    # 1)
    df.loc[:, 'scaled_iter'] = (df['iter'] - df['iter'].min()) / \
        (df['iter'].max() - df['iter'].min()) * 99 + 1

    # 2)
    df.loc[:, 'trPovertrAtA'] = df['trP'] / (df['froA'] * df['froA'])

    # 3)
    # Find rho values that give scaled number of iterations between 1 and 2
    rho_values = df.loc[(df['scaled_iter'] <= 5.0)].rho.values

    # Compute maximum and minimum values
    df.loc[:, 'rho_min'] = rho_values.min()
    df.loc[:, 'rho_max'] = rho_values.max()

    # if rho_values.min() == rho_values.max():
    #     print("[r_min, r_max] = [%.4e, %.4e]" %
    #           (rho_values.min(), rho_values.max()))
    #     import ipdb; ipdb.set_trace()

    return df


def get_grid_data(x, y, z, resX=100, resY=100):
    "Convert 3 column data to matplotlib grid"
    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')
    # X, Y = np.meshgrid(xi, yi)

    return xi, yi, zi


'''
Main script
'''

# Read full results
res = pd.read_csv('results/results_full.csv')

# Select problems not saturated at max number of iterations
res = res.loc[(res['iter'] < 2499)]

# Assign group headers
group_headers = ['seed', 'name']

print("Compute performance ratio for all problems and bounds for rho")

# Activate tqdm to see progress
tqdm.pandas()

# Assign performance and ratio to samples
problems = res.groupby(group_headers)
res_p = problems.progress_apply(get_ratio_and_bounds)
problems_p = res_p.groupby(group_headers)

n_problems = len(problems_p.groups)
print("\nTotal number of problems: %i" % n_problems)

'''
Construct problem
'''
# Number of parameters in alpha [alpha_0, alpha_1, alpha_2]
n_params = 5

# Data matrix A
A = spa.csc_matrix((0, n_params))

# rho bounds
v_l = np.empty((0))
v_u = np.empty((0))


print("Constructing data matrix A and bounds l, u")
for _, problem in tqdm(problems_p):
    
    n = problem['n'].iloc[0]
    m = problem['m'].iloc[0]
    trP = problem['trP'].iloc[0]
    trAtA = problem['froA'].iloc[0] ** 2

    # Create row of A matrix
    A_temp = np.array([1., np.log(n), np.log(m), np.log(trP), np.log(trAtA)])

    # Add row to matrix A
    A = spa.vstack((A, spa.csc_matrix(A_temp)), 'csc')

    # Add bounds on v
    l = np.log(problem['rho_min'].iloc[0])
    u = np.log(problem['rho_max'].iloc[0])
    v_l = np.append(v_l, l)
    v_u = np.append(v_u, u)


# Define CVXPY problem
alpha = cvxpy.Variable(n_params)
v = cvxpy.Variable(n_problems)

constraints = [v_l <= v, v <= v_u]
cost = cvxpy.norm(A * alpha - v)

# DEBUG: try to add regularization on rho
# lambda_reg = 1e-04
# cost += -lambda_reg * cvxpy.sum_entries(v)

objective = cvxpy.Minimize(cost)

problem = cvxpy.Problem(objective, constraints)

# Solve problem
problem.solve(solver=cvxpy.GUROBI, verbose=True)


'''
Create contour plot from 3D scatter plot with rho, ratio, scaled iterations
'''

# Get grid data
xi, yi, zi = get_grid_data(res_p['trPovertrAtA'],
                           res_p['rho'],
                           res_p['scaled_iter'])


# levels = [1., 3., 6., 10., 15., 20., 100.]
levels = [1., 2., 4., 6., 8., 10., 20., 100.]

# use here 256 instead of len(levels)-1 becuase
# as it's mentioned in the documentation for the
# colormaps, the default colormaps use 256 colors in their
# definition: print(plt.cm.jet.N) for example
norm = mc.BoundaryNorm(levels, 256)
# norm = None

# Try lognorm
# norm = mc.LogNorm(vmin=res_p['scaled_iter'].min(),
#                   vmax=res_p['scaled_iter'].max())


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
n_fit_points = 100

# Get learned alpha
alpha_fit = np.asarray(alpha.value).flatten()

# Get beta after removing logarithm
beta_fit = np.array([np.exp(alpha_fit[0]),
                     alpha_fit[1], alpha_fit[2],
                     alpha_fit[3], alpha_fit[4]])

# Get fit for every point
print("Fit rho through every point")
rho_fit = np.zeros(n_problems)
ratio_fit = np.zeros(n_problems)
i = 0
for _, problem in tqdm(problems_p):
    n = problem['n'].iloc[0]
    m = problem['m'].iloc[0]
    trP = problem['trP'].iloc[0]
    trAtA = problem['froA'].iloc[0] ** 2

    ratio_fit[i] = problem['trPovertrAtA'].iloc[0]

    rho_fit[i] = beta_fit[0] * \
            (n ** beta_fit[1]) * \
            (m ** beta_fit[2]) * \
            (trP ** beta_fit[3]) * \
            (trAtA ** beta_fit[4])

    i += 1

# Sort fit vectors
sort_fit_idx = np.argsort(ratio_fit)
ratio_fit = ratio_fit[sort_fit_idx]
rho_fit = rho_fit[sort_fit_idx]

# Pick only some elements of the picked vectors
interval_slice = 5
ratio_fit = ratio_fit[::interval_slice]
rho_fit = rho_fit[::interval_slice]

# Plot vectors
plt.plot(ratio_fit, rho_fit)


# plt.xscale('linear')
# plt.yscale('linear')
# plt.xscale('log')
# plt.yscale('log')

plt.show(block=False)
plt.savefig('behavior.pdf')
