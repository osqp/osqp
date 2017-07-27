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

# MAX_MIN_ITER = 0

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

    # 2)
    df.loc[:, 'trPovertrAtA'] = df['trP'] / (df['froA'] * df['froA'])

    # 3)
    df.loc[:, 'res_ratio'] = df['pri_res'] / df['dua_res']

    # 4)
    # Find rho values that give scaled number of iterations between certain
    # values
    rho_values = df.loc[(df['scaled_iter'] <= 2)].rho.values

    if len(rho_values) == 0:
        print('Problem with 0 rho_values')
        import ipdb; ipdb.set_trace()
        return None   # Too small problem description.

    # 5) Compute maximum and minimum values
    df.loc[:, 'rho_min'] = rho_values.min()
    df.loc[:, 'rho_max'] = rho_values.max()


    # 6) Compute best value of rho
    df.loc[:, 'best_rho'] = df.loc[(df['scaled_iter'] == 1)].rho.values.mean()

    # 7) Rho ratio
    df.loc[:, 'rho_ratio'] = df['best_rho']/df['rho']

    # DEBUG: Check max_min iter to see which problem gave the maximum number
    # of iterations
    # global MAX_MIN_ITER
    # if df['iter'].min() > MAX_MIN_ITER:
    #     MAX_MIN_ITER = df['iter'].min()

    if df['iter'].min() > 300:
        print("Bad problem) name = %s, best_rho = %.2e" %
              (df['name'].iloc[0], df['best_rho'].iloc[0]))

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

# Load results
prob_names = ['basis_pursuit',
              'huber_fit',
              'lasso',
              'nonneg_l2',
              #   'portfolio',
              'lp',
              'svm'
              ]

res_list = []
for prob_name in prob_names:
    res_temp = pd.read_csv(os.path.join('results', prob_name + '.csv'))
    res_list.append(res_temp)
res = pd.concat(res_list, ignore_index=True)

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
Construct fitting problem
'''
# Number of parameters in alpha [alpha_0, alpha_1, alpha_2, ..]
n_params = 4

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
    norm_q = problem['norm_q'].iloc[0]
    sigma = problem['sigma'].iloc[0]
    trAtA = problem['froA'].iloc[0] ** 2

    # A_temp = np.array([1.,
    #                    np.log(trP),
    #                    np.log(norm_q),
    #                    np.log(trAtA)])

    A_temp = np.array([1.,
                       trP,
                       norm_q,
                       trAtA])

    # Add row to matrix A
    A = spa.vstack((A, spa.csc_matrix(A_temp)), 'csc')

    # Add bounds on v
    # l = np.log(problem['rho_min'].iloc[0])
    # u = np.log(problem['rho_max'].iloc[0])
    l = problem['rho_min'].iloc[0]
    u = problem['rho_max'].iloc[0]
    v_l = np.append(v_l, l)
    v_u = np.append(v_u, u)

#
# Define CVXPY problem
alpha = cvxpy.Variable(n_params)
v = cvxpy.Variable(n_problems)

constraints = [v_l <= v, v <= v_u]

# Enforce elements to be nonnegative
# constraints += [v >= 0]

cost = cvxpy.norm(A * alpha - v)

objective = cvxpy.Minimize(cost)

problem = cvxpy.Problem(objective, constraints)

# Solve problem
print("Solving problem with CVXPY and MOSEK")
problem.solve(solver=cvxpy.MOSEK, verbose=True)
print("Solution status: %s" % problem.status)

# Get learned alpha
alpha_fit = np.asarray(alpha.value).flatten()

# beta_fit = np.array([np.exp(alpha_fit[0]),
#                      alpha_fit[1], alpha_fit[2],
#                      alpha_fit[3], alpha_fit[4], alpha_fit[5]])
beta_fit = alpha_fit

'''
Create plot: n_iter line and projected n_iter from rho_fit
'''
problems_idx = np.arange(n_problems)

# Initialize vectors for plotting
best_rhos = np.zeros(n_problems)
rho_min = np.zeros(n_problems)
rho_max = np.zeros(n_problems)
fit_rhos = np.zeros(n_problems)
min_iter = np.zeros(n_problems)
fit_iter = np.zeros(n_problems)
n = np.zeros(n_problems)
m = np.zeros(n_problems)
trP = np.zeros(n_problems)
norm_q = np.zeros(n_problems)
sigma = np.zeros(n_problems)
trAtA = np.zeros(n_problems)

i = 0
print("Finding rho fit and projected number of iterations")
for _, problem in tqdm(problems_p):

    # Get best rho from data
    best_rhos[i] = problem['best_rho'].iloc[0]

    # Get minimum and maximum rho
    rho_min[i] = problem['rho_min'].iloc[0]
    rho_max[i] = problem['rho_max'].iloc[0]

    # Get minimum number of iterations from data
    min_iter[i] = problem['iter'].min()

    # Get fit rho
    n[i] = problem['n'].iloc[0]
    m[i] = problem['m'].iloc[0]
    trP[i] = problem['trP'].iloc[0]
    norm_q[i] = problem['norm_q'].iloc[0]
    sigma[i] = problem['sigma'].iloc[0]
    trAtA[i] = problem['froA'].iloc[0] ** 2

    # fit_rhos[i] = beta_fit[0] * \
    #     (n[i] ** beta_fit[1]) * \
    #     (m[i] ** beta_fit[2]) * \
    #     ((trP[i] + sigma[i] * n[i]) ** beta_fit[3]) * \
    #     (norm_q[i] ** beta_fit[4]) * \
    #     (trAtA[i] ** beta_fit[5])

    fit_rhos[i] = beta_fit[0] + \
        trP[i] * beta_fit[1] + \
        norm_q[i] * beta_fit[2] + \
        trAtA[i] * beta_fit[3]

    # Get interpolated number of iterations from fit rho
    f_interp_iter = interp1d(problem['rho'].values,
                             problem['iter'].values,
                             bounds_error=False)

    fit_iter[i] = f_interp_iter(fit_rhos[i])

    # Update index
    i += 1


# Extra (Remove NaN values)
# not_nan_idx = np.logical_not(np.isnan(fit_iter))
# min_iter_new = min_iter[not_nan_idx]
# fit_iter_new = fit_iter[not_nan_idx]
# fit_rhos_new = fit_rhos[not_nan_idx]
# best_rhos_new = best_rhos[not_nan_idx]

# Order vector of iters
idx_sort = np.argsort(min_iter)
min_iter = min_iter[idx_sort]
fit_iter = fit_iter[idx_sort]

# Order vector of rhos
idx_sort = np.argsort(best_rhos)
best_rhos = best_rhos[idx_sort]
rho_min = rho_min[idx_sort]
rho_max = rho_max[idx_sort]
fit_rhos = fit_rhos[idx_sort]
n = n[idx_sort]
m = m[idx_sort]
norm_q = norm_q[idx_sort]
trP = trP[idx_sort]
trAtA = trAtA[idx_sort]


'''
Create actual plots
'''
# Fit rho
fig1 = plt.figure(1)
ax = plt.subplot(1, 1, 1)
ax.plot(best_rhos, label='Best rho')
ax.plot(rho_min, label='rho min', color='k', ls='-.', linewidth=.5)
ax.plot(rho_max, label='rho max', color='k', ls='-.', linewidth=.5)
ax.plot(fit_rhos, label='Fit rho')
# ax.plot(n, label='n')
# ax.plot(norm_q, label='norm_q')
# ax.plot(trP, label='trP')
# ax.plot(trP + norm_q, label='trP + norm_q')
# ax.plot(trAtA, label='trAtA')
plt.yscale('log')
plt.legend()
plt.grid()
plt.show(block=False)
plt.savefig('comparison_rho_fit.pdf')


# Fit iters
fig2 = plt.figure(2)
ax = plt.subplot(1, 1, 1)
ax.plot(min_iter, label='Min iter')
ax.plot(fit_iter, label='Fit iter')
plt.yscale('log')
plt.legend()
plt.grid()
plt.show(block=False)
plt.savefig('comparison_iter_fit.pdf')
