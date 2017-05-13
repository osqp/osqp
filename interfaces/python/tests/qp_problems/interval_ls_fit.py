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
        1) scaled number of iterations in [0, 100]
        2) ratio tr(P)/tr(A'A)
        3) lower and upper bounds for rho (between 0 and 10)
    """

    # 1)
    df.loc[:, 'scaled_iter'] = (df['iter'] - df['iter'].min()) / \
        (df['iter'].max() - df['iter'].min()) * 100

    # 2)
    df.loc[:, 'trPovertrAtA'] = df['trP'] / (df['froA'] * df['froA'])

    # 3)
    # Find rho values that give scaled number of iterations between 0 and 10
    rho_values = df.loc[(df['scaled_iter'] <= .1)].rho.values

    # Compute maximum and minimum values
    df.loc[:, 'rho_min'] = rho_values.min()
    df.loc[:, 'rho_max'] = rho_values.max()

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
res = res.loc[(res['iter'] < 1000)]

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
n_params = 1  # Number of parameters in beta

# Data matrix A
A = spa.csc_matrix((0, n_params))

# rho bounds
rho_l = np.empty((0))
rho_u = np.empty((0))

# Store ratios vector for later plotting
ratio_vec = np.empty((0))


print("Constructing data matrix A and bounds l, u")
for _, problem in tqdm(problems_p):
    ratio = problem['trPovertrAtA'].iloc[0]

    ratio_vec = np.append(ratio_vec, ratio)

    # Create row of A matrix
    A_temp = np.zeros(n_params)
    for i in range(n_params):
        A_temp[i] = ratio ** (i + 1)

    # Add row to matrix A
    A = spa.vstack((A, spa.csc_matrix(A_temp)), 'csc')

    # Add bounds on rho
    l = problem['rho_min'].iloc[0]
    u = problem['rho_max'].iloc[0]
    rho_l = np.append(rho_l, l)
    rho_u = np.append(rho_u, u)


# Define CVXPY problem
x = cvxpy.Variable(n_params)
rho = cvxpy.Variable(n_problems)

constraints = [rho_l <= rho, rho <= rho_u]
objective = cvxpy.Minimize(cvxpy.norm(A * x - rho))

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
n_fit_points = 100
x_fit = np.asarray(x.value).flatten()
ratio_vec = np.linspace(0, 2., n_fit_points)
rho_fit = np.zeros(n_fit_points)

for i in range(n_fit_points):
    a = np.zeros(n_params)
    for j in range(n_params):
        a[j] = ratio_vec[i] ** (j + 1)
    rho_fit[i] = a.dot(x_fit)


plt.plot(ratio_vec, rho_fit)

plt.show(block=False)
plt.savefig('behavior.pdf')
