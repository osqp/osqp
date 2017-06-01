# For plotting
import matplotlib.colors as mc
from scipy.interpolate import griddata
import utils.plotting as plotting
import matplotlib.pyplot as plt

import numpy as np
import scipy.sparse as spa
import os

# Dataframes
import pandas as pd
from tqdm import tqdm

# Formulate problem
import cvxpy

MAX_MIN_ITER = 0

def get_ratio_and_bounds(df):
    """
    Compute (relative to the current group)
        1) scaled number of iterations in [1, 100]
        2) ratio tr(P)/tr(A'A)
        3) ratio pri_res/dua_res 
        4) lower and upper bounds for rho (between 0 and 3)
        5) Compute maximum and minimum values of rho
    """

    # 1)
    df.loc[:, 'scaled_iter'] = (df['iter'] - df['iter'].min()) / \
        (df['iter'].max() - df['iter'].min()) * 99 + 1

    # DEBUG: Check max_min iter to see which problem gave the maximum number
    # of iterations

    global MAX_MIN_ITER
    if df['iter'].min() > MAX_MIN_ITER:
        MAX_MIN_ITER = df['iter'].min()

    # 2)
    df.loc[:, 'trPovertrAtA'] = df['trP'] / (df['froA'] * df['froA'])

    # 3)
    df.loc[:, 'res_ratio'] = df['pri_res'] / df['dua_res']

    # 4)
    # Find rho values that give scaled number of iterations between certain
    # values 
    rho_values = df.loc[(df['scaled_iter'] <= 1.5)].rho.values

    if len(rho_values) == 0:
        print('Problem with 0 rho_values')
        import ipdb; ipdb.set_trace()
        return None   # Too small problem description. 
                      # Probably haven't fnished
                      # computations yet

    # 5) Compute maximum and minimum values
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

# Load individual problems
#  prob_names = ['basis_pursuit', 'huber_fit', 'lasso', 'lp', 'nonneg_l2',
#                'portfolio', 'svm']
# Remove 'svm'
prob_names = ['basis_pursuit', 'huber_fit', 'lasso', 'nonneg_l2',
              'portfolio', 'lp']
# Only huber_fit
#  prob_names = ['huber_fit']

res_list = []
for prob_name in prob_names:
    res_temp = pd.read_csv(os.path.join('results', prob_name + '.csv')) 
    res_list.append(res_temp)
res = pd.concat(res_list, ignore_index=True)

#  import ipdb; ipdb.set_trace()
# Read full results
#  res = pd.read_csv('results/results_full.csv')

# Select problems not saturated at max number of iterations
res = res.loc[(res['iter'] < 2000)]

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
# Number of parameters in alpha [alpha_0, alpha_1, alpha_2, ..]
n_params = 3

# Data matrix A
A = spa.csc_matrix((0, n_params))

# rho bounds
v_l = np.empty((0))
v_u = np.empty((0))


print("Constructing data matrix A and bounds l, u")

for _, problem in tqdm(problems_p):

    n = problem['n'].iloc[0]
    m = problem['m'].iloc[0]
    sigma = problem['sigma'].iloc[0]
    trP = problem['trP'].iloc[0]

    trP_plus_sigma_n_over_n = (trP + sigma * n)/n


    trAtA_over_m = (problem['froA'].iloc[0] ** 2)/m

    #  if trP < 1e-06:
    #      print("trP = 0")
    #      import ipdb; ipdb.set_trace()
    #  logtrP = np.maximum(-1e06, np.log(trP))

    #  logtrAtA = np.maximum(-1e06, np.log(trAtA))
    #  logtrAtA = np.log(trAtA)

    # Create row of A matrix
    #  A_temp = np.array([1., np.log(n), np.log(m), np.log(trAtA)])
    A_temp = np.array([1.,
                       np.log(trP_plus_sigma_n_over_n), 
                       np.log(trAtA_over_m)])

    # Add row to matrix A
    A = spa.vstack((A, spa.csc_matrix(A_temp)), 'csc')

    # Add bounds on v
    l = np.log(problem['rho_min'].iloc[0])
    u = np.log(problem['rho_max'].iloc[0])
    v_l = np.append(v_l, l)
    v_u = np.append(v_u, u)

#
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

# Get learned alpha
alpha_fit = np.asarray(alpha.value).flatten()

# Get beta after removing logarithm
#  beta_fit = np.array([np.exp(alpha_fit[0]),
#                       alpha_fit[1], alpha_fit[2], alpha_fit[3]])
beta_fit = np.array([np.exp(alpha_fit[0]),
                     alpha_fit[1], alpha_fit[2]])



#  '''
#  Compute ratio residuals
#  '''
# stats for 1.5 <= rho <= 2
#  res_p.loc[((res_p['rho'] <= 2.) & (res_p['rho'] >= 1.5))][['res_ratio', 'iter']]

'''
Create contour plot from 3D scatter plot with rho, ratio, scaled iterations
'''
x_axis = np.zeros(len(res_p))
rho_values = np.zeros(len(res_p))
scaled_iter = np.zeros(len(res_p))

i = 0
for _, problem in tqdm(res_p.iterrows()):
    n = problem['n']
    m = problem['m']
    trP = problem['trP']
    sigma = problem['sigma']


    trAtA = problem['froA'] ** 2

    #  x_axis_fit[i] = beta_fit[0] * \
    #      (n ** beta_fit[1]) * \
    #      (m ** beta_fit[2]) * \
    #      (trAtA ** beta_fit[3])

    x_axis[i] = ((trP + sigma * n) / n) / (trAtA / m)

    #  x_axis_fit[i] = beta_fit[0] * \
    #      (n ** beta_fit[1]) * \
    #      (trAtA ** beta_fit[2])

    rho_values[i] = problem['rho']
    scaled_iter[i] = problem['scaled_iter']
    i += 1


xi, yi, zi = get_grid_data(np.log10(x_axis),
                           np.log10(rho_values),
                           scaled_iter)

# Revert logarithm
xi = np.power(10, xi)
yi = np.power(10, yi)


levels = [1., 3., 6., 10., 15., 20., 100.]
#  levels = [1., 2., 4., 6., 8., 10., 20., 100.]
#  levels = [1., 10., 20., 30., 40., 50., 60., 80., 90., 100.]

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
#  ax.set_xlabel(r'$\frac{\mathrm{tr} (P) + \sigma n}{\mathrm{tr}(A^{T}A)}$')
#  ax.set_xlabel(r'$i$')
ax.set_title(r'Scaled number of iterations')
#  ax.set_xlim(0., 2.)
plt.colorbar()
plt.tight_layout()

'''
Plot fit line on the graph
'''
n_fit_points = 100


# Get fit for every point
print("Fit rho through every point")
rho_fit = np.zeros(n_problems)
ratio_fit = np.zeros(n_problems)
i = 0
for _, problem in tqdm(problems_p):
    n = problem['n'].iloc[0]
    trP = problem['trP'].iloc[0]
    trAtA = problem['froA'].iloc[0] ** 2
    sigma = problem['sigma'].iloc[0]

    ratio_fit[i] = ((trP + sigma * n)/n) / (trAtA/m)

    #  rho_fit[i] = beta_fit[0] * \
    #          (n ** beta_fit[1]) * \
    #          (trP ** beta_fit[3]) * \
    #          (trAtA ** beta_fit[4])

    rho_fit[i] = beta_fit[0] * \
        (((trP + sigma * n)/n) ** beta_fit[1]) * \
        ((trAtA/m) ** beta_fit[2])
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

#
#  plt.xscale('linear')
# plt.yscale('linear')
plt.xscale('log')
plt.yscale('log')

plt.show(block=False)
plt.savefig('behavior.pdf')



#  '''
#  Create contour plot from 3D scatter plot with rho,
#  ((trP + sigma * n) / n) / (trAtA / m),
#  scaled iterations
#  '''
#  x_axis = np.zeros(len(res_p))
#  rho_values = np.zeros(len(res_p))
#  scaled_iter = np.zeros(len(res_p))
#
#  i = 0
#  for _, problem in tqdm(res_p.iterrows()):
#      n = problem['n']
#      m = problem['m']
#      sigma = problem['sigma']
#      trP = problem['trP']
#      trAtA = problem['froA'] ** 2
#
#      x_axis[i] = ((trP + sigma * n) / n) / (trAtA / m)
#
#      rho_values[i] = problem['rho']
#      scaled_iter[i] = problem['scaled_iter']
#      i += 1
#
#
#  xi, yi, zi = get_grid_data(np.log10(x_axis),
#                             np.log10(rho_values),
#                             scaled_iter)
#
#  # Revert logarithm
#  xi = np.power(10, xi)
#  yi = np.power(10, yi)
#
#
#  #  levels = [1., 3., 6., 10., 15., 20., 100.]
#  levels = [1., 1.5, 2., 2.5, 3., 6., 10., 15., 20., 100.]
#  norm = mc.BoundaryNorm(levels, 256)
#
#  ax = plotting.create_figure(0.9)
#  plt.contour(xi, yi, zi, levels=levels, norm=norm, cmap=plt.cm.viridis_r)
#  plt.contourf(xi, yi, zi, levels=levels, norm=norm, cmap=plt.cm.viridis_r)
#  ax.set_ylabel(r'$\rho$')
#  #  ax.set_xlabel(r'$\frac{\mathrm{tr} (P) + \sigma n}{\mathrm{tr}(A^{T}A)}$')
#  ax.set_title(r'Scaled number of iterations')
#  #  ax.set_xlim(0., 2.)
#  plt.colorbar()
#  plt.tight_layout()
#
#
#  #  plt.xscale('log')
#  plt.yscale('log')
#
#  plt.show(block=False)
#  plt.savefig('behavior_complex_ratio.pdf')
#


#  '''
#  Create contour plot from 3D scatter plot with rho, n/m, scaled iterations
#  '''
#  x_axis = np.zeros(len(res_p))
#  rho_values = np.zeros(len(res_p))
#  scaled_iter = np.zeros(len(res_p))
#
#  i = 0
#  for _, problem in tqdm(res_p.iterrows()):
#      n = problem['n']
#      m = problem['m']
#      x_axis[i] = n / m
#
#      rho_values[i] = problem['rho']
#      scaled_iter[i] = problem['scaled_iter']
#      i += 1
#
#
#  xi, yi, zi = get_grid_data(np.log10(x_axis),
#                             np.log10(rho_values),
#                             scaled_iter)
#
#  # Revert logarithm
#  xi = np.power(10, xi)
#  yi = np.power(10, yi)
#
#
#  #  levels = [1., 3., 6., 10., 15., 20., 100.]
#  levels = [1., 1.5, 2., 2.5, 3., 6., 10., 15., 20., 100.]
#  norm = mc.BoundaryNorm(levels, 256)
#
#  ax = plotting.create_figure(0.9)
#  plt.contour(xi, yi, zi, levels=levels, norm=norm, cmap=plt.cm.viridis_r)
#  plt.contourf(xi, yi, zi, levels=levels, norm=norm, cmap=plt.cm.viridis_r)
#  ax.set_ylabel(r'$\rho$')
#  ax.set_xlabel(r'$\frac{n}{m}$')
#  ax.set_title(r'Scaled number of iterations')
#  #  ax.set_xlim(0., 2.)
#  plt.colorbar()
#  plt.tight_layout()
#
#
#  plt.xscale('log')
#  plt.yscale('log')
#
#  plt.show(block=False)
#  plt.savefig('behavior_n_over_m_ratio.pdf')
#



#  '''
#  Create contour plot from 3D scatter plot with rho, residual ratio, scaled iterations
#  '''
#  xi, yi, zi = get_grid_data(np.log10(res_p['res_ratio'].values),
#                             np.log10(res_p['rho'].values),
#                             res_p['scaled_iter'].values)
#
#  # Revert logarithm
#  xi = np.power(10, xi)
#  yi = np.power(10, yi)
#
#  # Set levels and norm
#  levels = [1., 2., 4., 6., 8., 10., 20., 100.]
#  norm = mc.BoundaryNorm(levels, 256)
#
#
#
#  ax = plotting.create_figure(0.9)
#  plt.contour(xi, yi, zi, levels=levels, norm=norm, cmap=plt.cm.viridis_r)
#  plt.contourf(xi, yi, zi, levels=levels, norm=norm, cmap=plt.cm.viridis_r)
#  ax.set_ylabel(r'$\rho$')
#  ax.set_xlabel(r'$\frac{\|r_{\mathrm{pri}}\|_{\infty}}{\|r_{\mathrm{dua}}\|_{\infty}}$')
#  ax.set_title(r'Scaled number of iterations')
#  plt.colorbar()
#  plt.tight_layout()
#
#  # plt.xscale('linear')
#  # plt.yscale('linear')
#  plt.xscale('log')
#  plt.yscale('log')
#
#  plt.show(block=False)
#  plt.savefig('behavior_res_ratio.pdf')
