# For plotting
import matplotlib.colors as mc
from scipy.interpolate import griddata
#  import utils.plotting as plotting
import matplotlib.pyplot as plt

import numpy as np
import scipy.sparse as spa
import os


# Interpolate number of iterations obtained
from scipy.interpolate import interp1d


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
        6) Compute best value of rho
        7) Compute all the relative values of rho: best_rho / rho
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
    rho_values = df.loc[(df['scaled_iter'] <= 1)].rho.values

    if len(rho_values) == 0:
        print('Problem with 0 rho_values')
        import ipdb; ipdb.set_trace()
        return None   # Too small problem description. 
                      # Probably haven't fnished
                      # computations yet

    # 5) Compute maximum and minimum values
    df.loc[:, 'rho_min'] = rho_values.min()
    df.loc[:, 'rho_max'] = rho_values.max()


    # 6) Compute best value of rho
    df.loc[:, 'best_rho'] = df.loc[(df['scaled_iter'] == 1)].rho.values.mean()

    # 7) Rho ratio
    df.loc[:, 'rho_ratio'] = df['best_rho']/df['rho']

    return df


#  def get_grid_data(x, y, z, resX=100, resY=100):
#      "Convert 3 column data to matplotlib grid"
#      xi = np.linspace(min(x), max(x), resX)
#      yi = np.linspace(min(y), max(y), resY)
#      zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')
#      # X, Y = np.meshgrid(xi, yi)
#
#      return xi, yi, zi


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
    sigma = problem['sigma'].iloc[0]
    trAtA = problem['froA'].iloc[0] ** 2

    A_temp = np.array([1.,
                       np.log(n),
                       np.log(m),
                       np.log(trP + sigma * n),
                       np.log(trAtA)])

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

objective = cvxpy.Minimize(cost)

problem = cvxpy.Problem(objective, constraints)

# Solve problem
problem.solve(solver=cvxpy.GUROBI, verbose=True)

# Get learned alpha
alpha_fit = np.asarray(alpha.value).flatten()

beta_fit = np.array([np.exp(alpha_fit[0]),
                     alpha_fit[1], alpha_fit[2],
                     alpha_fit[3], alpha_fit[4]])


'''
Create plot: n_iter line and projected n_iter from rho_fit
'''
problems_idx = np.arange(n_problems)

# Initialize vectors for plotting
best_rhos = np.zeros(n_problems)
fit_rhos = np.zeros(n_problems)
min_iter = np.zeros(n_problems)
fit_iter = np.zeros(n_problems)

i = 0
print("Finding rho fit and projected number of iterations")
for _, problem in tqdm(problems_p):

    # Get best rho from data
    best_rhos[i] = problem['best_rho'].iloc[0]    

    # Get minimum number of iterations from data
    min_iter[i] = problem['iter'].min()

    # Get fit rho
    n = problem['n'].iloc[0]
    m = problem['m'].iloc[0]
    trP = problem['trP'].iloc[0]
    sigma = problem['sigma'].iloc[0]
    trAtA = problem['froA'].iloc[0] ** 2

    fit_rhos[i] = beta_fit[0] * \
        (n ** beta_fit[1]) * \
        (m ** beta_fit[2]) * \
        ((trP + sigma * n) ** beta_fit[3]) * \
        (trAtA ** beta_fit[4])

    # Get interpolated number of iterations from fit rho
    f_interp_iter = interp1d(problem['rho'].values,
                             problem['iter'].values,
                             bounds_error=False)

    fit_iter[i] = f_interp_iter(fit_rhos[i])

    # Update index
    i += 1


# Extra (Remove NaN values)
not_nan_idx = np.logical_not(np.isnan(fit_iter))
min_iter_new = min_iter[not_nan_idx]
fit_iter_new = fit_iter[not_nan_idx]
fit_rhos_new = fit_rhos[not_nan_idx]
best_rhos_new = best_rhos[not_nan_idx]

# Order vector of iters 
idx_sort = np.argsort(min_iter_new)
min_iter_new = min_iter_new[idx_sort]
fit_iter_new = fit_iter_new[idx_sort]

# Order vector of rhos 
idx_sort = np.argsort(best_rhos_new)
best_rhos_new = best_rhos_new[idx_sort]
fit_rhos_new = fit_rhos_new[idx_sort]





'''
Create actual plots
'''
# Fit rho 
fig, ax = plt.subplots()
ax.plot(best_rhos_new, label='Best rho')
ax.plot(fit_rhos_new, label='Fit rho')
plt.yscale('log')
plt.legend()
plt.grid()
plt.show(block=False)
plt.savefig('comparison_rho_fit.pdf')


# Fit iters 
fig, ax = plt.subplots()
ax.plot(min_iter_new, label='Min iter')
ax.plot(fit_iter_new, label='Fit iter')
plt.yscale('log')
plt.legend()
plt.grid()
plt.show(block=False)
plt.savefig('comparison_iter_fit.pdf')




#
#  '''
#  Create contour plot from 3D scatter plot with rho, ratio, scaled iterations
#  '''
#  x_axis = np.zeros(len(res_p))
#  rho_values = np.zeros(len(res_p))
#  scaled_iter = np.zeros(len(res_p))
#
#  i = 0
#  for _, problem in tqdm(res_p.iterrows()):
#      n = problem['n']
#      m = problem['m']
#
#      min_n_m = np.minimum(n, m)
#
#      x_axis[i] = min_n_m / m
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
#  levels = [1., 3., 6., 10., 15., 20., 100.]
#
#  # use here 256 instead of len(levels)-1 becuase
#  # as it's mentioned in the documentation for the
#  # colormaps, the default colormaps use 256 colors in their
#  # definition: print(plt.cm.jet.N) for example
#  norm = mc.BoundaryNorm(levels, 256)
#  # norm = None
#
#
#  ax = plotting.create_figure(0.9)
#  plt.contour(xi, yi, zi, levels=levels, norm=norm, cmap=plt.cm.viridis_r)
#  plt.contourf(xi, yi, zi, levels=levels, norm=norm, cmap=plt.cm.viridis_r)
#  ax.set_ylabel(r'$\rho$')
#  #  ax.set_xlabel(r'$\frac{\mathrm{tr} (P) + \sigma n}{\mathrm{tr}(A^{T}A)}$')
#  ax.set_xlabel(r'$\frac{\min (n, m)}{m}$')
#  ax.set_title(r'Scaled number of iterations')
#  plt.colorbar()
#  plt.tight_layout()
#
#  '''
#  Plot fit line on the graph
#  '''
#  n_fit_points = 100
#
#
#  # Get fit for every point
#  print("Fit rho through every point")
#  rho_fit = np.zeros(n_problems)
#  ratio_fit = np.zeros(n_problems)
#  i = 0
#  for _, problem in tqdm(problems_p):
#      n = problem['n'].iloc[0]
#      m = problem['m'].iloc[0]
#
#      min_n_m = np.minimum(n, m)
#
#      ratio_fit[i] = min_n_m / m
#
#
#      rho_fit[i] = beta_fit[0] * \
#          (min_n_m ** beta_fit[1]) * \
#          (m ** beta_fit[2])
#      i += 1
#
#  # Sort fit vectors
#  sort_fit_idx = np.argsort(ratio_fit)
#  ratio_fit = ratio_fit[sort_fit_idx]
#  rho_fit = rho_fit[sort_fit_idx]
#
#  # Pick only some elements of the picked vectors
#  interval_slice = 5
#  ratio_fit = ratio_fit[::interval_slice]
#  rho_fit = rho_fit[::interval_slice]
#
#  # Plot vectors
#  plt.plot(ratio_fit, rho_fit)
#
#  #
#  #  plt.xscale('linear')
#  # plt.yscale('linear')
#  plt.xscale('log')
#  plt.yscale('log')
#
#  plt.show(block=False)
#  plt.savefig('behavior.pdf')
#
#
#
#  '''
#  Plot scatter plot of rho ratio
#  '''
#  #  TODO: Continue from here! Make scatter plot visible!
#
#  ax = plotting.create_figure(0.9)
#  plt.scatter(res_p['res_ratio'], res_p['rho_ratio'])
#
#  ax.set_ylabel(r'$\frac{\rho^{\star}}{\rho}$')
#  ax.set_xlabel(r'$\frac{\|r_{\mathrm{pri}}\|_{\infty}}{\|r_{\mathrm{dua}}\|_{\infty}}$')
#  plt.xscale('log')
#  plt.yscale('log')
#  plt.tight_layout()
#  ax.set_xlim(1e-09, 1e09)
#  ax.set_ylim(1e-09, 1e09)
#  plt.savefig('behavior_scatter_rho_ratio.pdf')
#
#  #  '''
#  #  Create contour plot from 3D scatter plot with rho,
#  #  ((trP + sigma * n) / n) / (trAtA / m),
#  #  scaled iterations
#  #  '''
#  #  x_axis = np.zeros(len(res_p))
#  #  rho_values = np.zeros(len(res_p))
#  #  scaled_iter = np.zeros(len(res_p))
#  #
#  #  i = 0
#  #  for _, problem in tqdm(res_p.iterrows()):
#  #      n = problem['n']
#  #      m = problem['m']
#  #      sigma = problem['sigma']
#  #      trP = problem['trP']
#  #      trAtA = problem['froA'] ** 2
#  #
#  #      x_axis[i] = ((trP + sigma * n) / n) / (trAtA / m)
#  #
#  #      rho_values[i] = problem['rho']
#  #      scaled_iter[i] = problem['scaled_iter']
#  #      i += 1
#  #
#  #
#  #  xi, yi, zi = get_grid_data(np.log10(x_axis),
#  #                             np.log10(rho_values),
#  #                             scaled_iter)
#  #
#  #  # Revert logarithm
#  #  xi = np.power(10, xi)
#  #  yi = np.power(10, yi)
#  #
#  #
#  #  #  levels = [1., 3., 6., 10., 15., 20., 100.]
#  #  levels = [1., 1.5, 2., 2.5, 3., 6., 10., 15., 20., 100.]
#  #  norm = mc.BoundaryNorm(levels, 256)
#  #
#  #  ax = plotting.create_figure(0.9)
#  #  plt.contour(xi, yi, zi, levels=levels, norm=norm, cmap=plt.cm.viridis_r)
#  #  plt.contourf(xi, yi, zi, levels=levels, norm=norm, cmap=plt.cm.viridis_r)
#  #  ax.set_ylabel(r'$\rho$')
#  #  #  ax.set_xlabel(r'$\frac{\mathrm{tr} (P) + \sigma n}{\mathrm{tr}(A^{T}A)}$')
#  #  ax.set_title(r'Scaled number of iterations')
#  #  #  ax.set_xlim(0., 2.)
#  #  plt.colorbar()
#  #  plt.tight_layout()
#  #
#  #
#  #  #  plt.xscale('log')
#  #  plt.yscale('log')
#  #
#  #  plt.show(block=False)
#  #  plt.savefig('behavior_complex_ratio.pdf')
#  #
#
#
#  #  '''
#  #  Create contour plot from 3D scatter plot with rho, n/m, scaled iterations
#  #  '''
#  #  x_axis = np.zeros(len(res_p))
#  #  rho_values = np.zeros(len(res_p))
#  #  scaled_iter = np.zeros(len(res_p))
#  #
#  #  i = 0
#  #  for _, problem in tqdm(res_p.iterrows()):
#  #      n = problem['n']
#  #      m = problem['m']
#  #      x_axis[i] = n / m
#  #
#  #      rho_values[i] = problem['rho']
#  #      scaled_iter[i] = problem['scaled_iter']
#  #      i += 1
#  #
#  #
#  #  xi, yi, zi = get_grid_data(np.log10(x_axis),
#  #                             np.log10(rho_values),
#  #                             scaled_iter)
#  #
#  #  # Revert logarithm
#  #  xi = np.power(10, xi)
#  #  yi = np.power(10, yi)
#  #
#  #
#  #  #  levels = [1., 3., 6., 10., 15., 20., 100.]
#  #  levels = [1., 1.5, 2., 2.5, 3., 6., 10., 15., 20., 100.]
#  #  norm = mc.BoundaryNorm(levels, 256)
#  #
#  #  ax = plotting.create_figure(0.9)
#  #  plt.contour(xi, yi, zi, levels=levels, norm=norm, cmap=plt.cm.viridis_r)
#  #  plt.contourf(xi, yi, zi, levels=levels, norm=norm, cmap=plt.cm.viridis_r)
#  #  ax.set_ylabel(r'$\rho$')
#  #  ax.set_xlabel(r'$\frac{n}{m}$')
#  #  ax.set_title(r'Scaled number of iterations')
#  #  #  ax.set_xlim(0., 2.)
#  #  plt.colorbar()
#  #  plt.tight_layout()
#  #
#  #
#  #  plt.xscale('log')
#  #  plt.yscale('log')
#  #
#  #  plt.show(block=False)
#  #  plt.savefig('behavior_n_over_m_ratio.pdf')
#  #
#
#
#
#  #  '''
#  #  Create contour plot from 3D scatter plot with rho, residual ratio, scaled iterations
#  #  '''
#  #  xi, yi, zi = get_grid_data(np.log10(res_p['res_ratio'].values),
#  #                             np.log10(res_p['rho'].values),
#  #                             res_p['scaled_iter'].values)
#  #
#  #  # Revert logarithm
#  #  xi = np.power(10, xi)
#  #  yi = np.power(10, yi)
#  #
#  #  # Set levels and norm
#  #  levels = [1., 2., 4., 6., 8., 10., 20., 100.]
#  #  norm = mc.BoundaryNorm(levels, 256)
#  #
#  #
#  #
#  #  ax = plotting.create_figure(0.9)
#  #  plt.contour(xi, yi, zi, levels=levels, norm=norm, cmap=plt.cm.viridis_r)
#  #  plt.contourf(xi, yi, zi, levels=levels, norm=norm, cmap=plt.cm.viridis_r)
#  #  ax.set_ylabel(r'$\rho$')
#  #  ax.set_xlabel(r'$\frac{\|r_{\mathrm{pri}}\|_{\infty}}{\|r_{\mathrm{dua}}\|_{\infty}}$')
#  #  ax.set_title(r'Scaled number of iterations')
#  #  plt.colorbar()
#  #  plt.tight_layout()
#  #
#  #  # plt.xscale('linear')
#  #  # plt.yscale('linear')
#  #  plt.xscale('log')
#  #  plt.yscale('log')
#  #
#  #  plt.show(block=False)
#  #  plt.savefig('behavior_res_ratio.pdf')
