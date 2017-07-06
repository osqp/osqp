# For plotting
import matplotlib.colors as mc
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
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


'''
Main script
'''

# Load individual problems
prob_names = [
              'basis_pursuit',
              'huber_fit',
              'lasso',
              'nonneg_l2',
              'portfolio',
              'svm',
              'lp'
            ]

res_list = []
for prob_name in prob_names:
    res_temp = pd.read_csv(os.path.join('results', prob_name + '.csv'))
    res_list.append(res_temp)
res = pd.concat(res_list, ignore_index=True)

import ipdb; ipdb.set_trace()
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

    norm_q = problem['norm_q'].iloc[0]
    avg_u_minus_l = problem['avg_u_minus_l'].iloc[0]

    A_temp = np.array([1.,
                       np.log(norm_q),
                       np.log(avg_u_minus_l)])

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

#  beta_fit = np.array([np.exp(alpha_fit[0]),
#                       alpha_fit[1], alpha_fit[2],
#                       alpha_fit[3], alpha_fit[4]])

beta_fit = np.array([np.exp(alpha_fit[0]),
                     alpha_fit[1], alpha_fit[2]])

#  '''
#  Fit power law starting from log-log result
#  '''
#
#
#  def powerlaw(x, beta0, beta1, beta2):
#      return beta0 * (x[0] ** beta1) * (x[1] ** beta2)
#
#
#  # Get xdata and ydata
#  xdata = np.zeros((n_params - 1, n_problems))
#  ydata = np.zeros(n_problems)
#
#  print("Construct xdata and ydata for nonlinear least squares fit")
#  i = 0
#  for _, problem in tqdm(problems_p):
#
#      n = problem['n'].iloc[0]
#      m = problem['m'].iloc[0]
#      xdata[:, i] = np.array([n, m])
#      ydata[i] = problem['best_rho'].iloc[0]
#
#      i += 1
#
#  # Perform curve fitting
#  popt, pcov = curve_fit(powerlaw, xdata, ydata, p0=beta_fit, sigma=np.abs(ydata))
#
#  # Assign new parameters
#  beta_fit_linear = beta_fit
#  beta_fit = popt
#


'''
Create plot: n_iter line and projected n_iter from rho_fit
'''
problems_idx = np.arange(n_problems)

# Initialize vectors for plotting
best_rhos = np.zeros(n_problems)
fit_rhos = np.zeros(n_problems)
min_iter = np.zeros(n_problems)
fit_iter = np.zeros(n_problems)
norm_q = np.zeros(n_problems)
avg_u_minus_l = np.zeros(n_problems)

# Get vector of xdata for final fitting


i = 0
print("Finding rho fit and projected number of iterations")
for _, problem in tqdm(problems_p):

    # Get best rho from data
    best_rhos[i] = problem['best_rho'].iloc[0]

    # Get minimum number of iterations from data
    min_iter[i] = problem['iter'].min()

    # Get fit rho
    norm_q[i] = problem['norm_q'].iloc[0]
    avg_u_minus_l[i] = problem['avg_u_minus_l'].iloc[0]


    fit_rhos[i] = beta_fit[0] * \
        (norm_q[i] ** beta_fit[1]) * \
        (avg_u_minus_l[i] ** beta_fit[2])


    # Get interpolated number of iterations from fit rho
    f_interp_iter = interp1d(problem['rho'].values,
                             problem['iter'].values,
                             bounds_error=False)

    fit_iter[i] = f_interp_iter(fit_rhos[i])

    # Update index
    i += 1




'''
Create actual plots
'''

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

# Order vector of norm_q
norm_q = norm_q[idx_sort]
avg_u_minus_l = avg_u_minus_l[idx_sort]

# Fit rho
fig, ax = plt.subplots()
ax.plot(best_rhos_new, label='Best rho')
ax.plot(fit_rhos_new, label='Fit rho')
plt.yscale('log')
plt.legend()
plt.grid()
plt.show(block=False)
# plt.savefig('comparison_rho_fit.pdf')


# Fit iters
fig, ax = plt.subplots()
ax.plot(min_iter_new, label='Min iter')
ax.plot(fit_iter_new, label='Fit iter')
plt.yscale('log')
plt.legend()
plt.grid()
plt.show(block=False)
# plt.savefig('comparison_iter_fit.pdf')

# Norm_q and best_rho
fig, ax = plt.subplots()
ax.plot(best_rhos_new, label='Best rho')
ax.plot(norm_q, label='||q||')
plt.legend()
plt.grid()
plt.show(block=False)
