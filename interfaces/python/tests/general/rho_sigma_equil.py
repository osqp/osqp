'''
Try different values of rho and sigma to check relative norms
between columns of KKT matrix
'''
import osqp
#  import osqppurepy as osqp
import scipy.sparse as spa
import scipy as sp
import numpy as np
import mathprogbasepy as mpbpy
from tqdm import tqdm

import matplotlib.colors as mc
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# Reset seed for reproducibility
sp.random.seed(3)


def get_grid_data(x, y, z, resX=500, resY=500):
    "Convert 3 column data to matplotlib grid"
    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')
    # X, Y = np.meshgrid(xi, yi)

    return xi, yi, zi


# Scaling
SCALING_REG = 1e-06


def scale_data(P, q, A, l, u, scaling=100):
    """
    Perform symmetric diagonal scaling via equilibration
    """
    (m, n) = A.shape

    # Initialize scaling
    d = np.ones(n + m)
    d_temp = np.ones(n + m)

    # Define reduced KKT matrix to scale
    KKT = spa.vstack([
          spa.hstack([P, A.T]),
          spa.hstack([A, spa.csc_matrix((m, m))])]).tocsc()

    # Iterate Scaling
    for i in range(scaling):
        for j in range(n + m):
            norm_col_j = np.linalg.norm(np.asarray(KKT[:, j].todense()), 
                                        np.inf)
            if norm_col_j > SCALING_REG:
                d_temp[j] = 1./(np.sqrt(norm_col_j))

        S_temp = spa.diags(d_temp)
        d = np.multiply(d, d_temp)
        KKT = S_temp.dot(KKT.dot(S_temp)) 

    # Obtain Scaler Matrices
    D = spa.diags(d[:n])
    if m == 0:
        # spa.diags() will throw an error if fed with an empty array
        E = spa.csc_matrix((0, 0))
    else:
        E = spa.diags(d[n:])

    # Scale problem Matrices
    P = D.dot(P.dot(D)).tocsc()
    A = E.dot(A.dot(D)).tocsc()
    q = D.dot(q)
    l = E.dot(l)
    u = E.dot(u)

    #  import ipdb; ipdb.set_trace()

    return P, q, A, l, u


def get_KKT_info(P, A, sigma, rho):
    '''
    Get

    1) Condition number: cond(KKT)

    2) Ratio: r = max_{i} || KKT_{i, :} || / min_{i} || KKT_{i, :} ||

    where || . || are the infinity norms and

          KKT = [P + sigma I   A'
                 A             - 1/rho I]
    
    NB: It computes the ratio for rows but it is equal to the columns
         one since KKT is symmetric
    '''
    # Get problem dimensions
    (m, n) = A.shape

    # Construct KKT matrix
    KKT = spa.vstack([
          spa.hstack([P + sigma * spa.eye(n), A.T]),
          spa.hstack([A, -1./rho * spa.eye(m)])]).todense()

    # Get ratio between columns and rows
    n_plus_m = n + m
    max_norm_rows = 0.0
    min_norm_rows = np.inf
    for j in range(n_plus_m):
        norm_row_j = np.linalg.norm(KKT[j, :], np.inf)
        max_norm_rows = np.maximum(norm_row_j,
                                   max_norm_rows)
        min_norm_rows = np.minimum(norm_row_j,
                                   min_norm_rows)

    # Compute ratio
    r = max_norm_rows / min_norm_rows

    # Get condition number
    cond_KKT = np.linalg.cond(KKT)

    # Return ratios
    return r, cond_KKT


'''
Generate QP
'''
n = 20
m = 30

# Constraints
random_scaling = spa.diags(np.power(10, np.random.randn(m)))
A = random_scaling.dot(spa.random(m, n, density=0.4, format='csc')).tocsc()
l = -random_scaling.dot(sp.rand(m))
u = random_scaling.dot(sp.rand(m))
# Cost function
random_scaling = spa.diags(np.power(10, np.random.randn(n)))
P = random_scaling.dot(spa.random(n, n, density=0.2)).tocsc()
P = P.dot(P.T).tocsc()
q = random_scaling.dot(sp.randn(n))


# Scale data as OSQP does
scaling = 15 
if scaling != 0:
    (P_sc, q_sc, A_sc, l_sc, u_sc) = (P, q, A, l, u)
else:
    P_sc, q_sc, A_sc, l_sc, u_sc = scale_data(P, q, A, l, u, 
                                              scaling=scaling)


# Iterate over rho and sigma values solving the problem
rho_min = 1e-07
rho_max = 1e07
n_rho = 50
sigma_min = 1e-07
sigma_max = 1e06
n_sigma = 50

rho_vec = np.logspace(np.log10(rho_min), np.log10(rho_max), num=n_rho)
sigma_vec = np.logspace(np.log10(sigma_min), np.log10(sigma_max), num=n_sigma)


# Preallocate number of iterations
n_iter = np.zeros((n_rho, n_sigma))

# Preallocate ratio
r_KKT = np.zeros((n_rho, n_sigma))

# Preallocate condition numbers
cond_KKT = np.zeros((n_rho, n_sigma))

# Preallocate ratio between residuals
r_res = np.zeros((n_rho, n_sigma))

# Preallocate values for plotting
rho_plot = np.zeros(n_rho * n_sigma)
sigma_plot = np.zeros(n_rho * n_sigma)
r_KKT_plot = np.zeros(n_rho * n_sigma)
r_res_plot = np.zeros(n_rho * n_sigma)
cond_KKT_plot = np.zeros(n_rho * n_sigma)
n_iter_plot = np.zeros(n_rho * n_sigma)
z_idx = 0

for i in tqdm(range(len(rho_vec))):
    for j in range(len(sigma_vec)):
        # Get ratios of norms for columns and rows
        r_KKT[i, j], cond_KKT[i, j] = get_KKT_info(P_sc, A_sc,
                                                   sigma_vec[j],
                                                   rho_vec[i])

        # Solve problem
        m = osqp.OSQP()
        m.setup(P, q, A, l, u,
                rho=rho_vec[i],
                sigma=sigma_vec[j],
                scaling=scaling,
                polish=False,
                verbose=False)
        res = m.solve()

        # Store number of iterations
        n_iter[i, j] = res.info.iter

        # Store ratio between residuals
        r_res[i, j] = res.info.pri_res / res.info.dua_res

        # Allocate values for plotting
        rho_plot[z_idx] = rho_vec[i]
        sigma_plot[z_idx] = sigma_vec[j]
        r_KKT_plot[z_idx] = r_KKT[i, j]
        r_res_plot[z_idx] = r_res[i, j]
        cond_KKT_plot[z_idx] = cond_KKT[i, j]
        n_iter_plot[z_idx] = n_iter[i, j]
        z_idx += 1



'''
Generate plot iterations
'''
# Get grid data (take logarithm of rho and sigma to have better gridding)
xi, yi, zi = get_grid_data(np.log10(rho_plot), np.log10(sigma_plot), n_iter_plot)

# Revert logarithm
xi = np.power(10, xi)
yi = np.power(10, yi)

#  low_levels = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50] 
#  levels = [n_iter_plot.min() * i for i in low_levels] 
#  levels.append(n_iter_plot.max())
levels_step = (n_iter_plot.max() - n_iter_plot.min())/10.
levels = np.arange(n_iter_plot.min(), n_iter_plot.max() + levels_step, 
                   levels_step)

# use here 256 instead of len(levels)-1 becuase
# as it's mentioned in the documentation for the
# colormaps, the default colormaps use 256 colors in their
# definition: print(plt.cm.jet.N) for example
norm = mc.BoundaryNorm(levels, 256)

# Generate plot
fig = plt.figure()
ax = fig.add_subplot(111)
plt.contour(xi, yi, zi, norm=norm, levels=levels)
plt.contourf(xi, yi, zi, norm=norm, levels=levels)
ax.set_ylabel(r'$\sigma$')
ax.set_xlabel(r'$\rho$')
ax.set_title(r'Number of iterations')
plt.colorbar()
plt.tight_layout()
ax.set_xscale('log')
ax.set_yscale('log')
plt.savefig('sigma_rho_n_iter.pdf')

'''
Generate plot ratio between norms
'''
# Get grid data
xi, yi, zi = get_grid_data(np.log10(rho_plot), np.log10(sigma_plot),
                           np.log10(r_KKT_plot))

# Revert logarithm
xi = np.power(10, xi)
yi = np.power(10, yi)
zi = np.power(10, zi)

norm = mc.LogNorm(vmin=zi.min(),
                  vmax=zi.max())

# Generate plot
fig = plt.figure()
ax = fig.add_subplot(111)
plt.contour(xi, yi, zi, norm=norm)
plt.contourf(xi, yi, zi, norm=norm)
ax.set_ylabel(r'$\sigma$')
ax.set_xlabel(r'$\rho$')
ax.set_title(r'Ratio max/min norm of KKT rows')
plt.colorbar()
plt.tight_layout()
ax.set_xscale('log')
ax.set_yscale('log')
plt.savefig('sigma_rho_ratio_norms_KKT.pdf')


'''
Generate plot ratio between residuals
'''
# Get grid data
xi, yi, zi = get_grid_data(np.log10(rho_plot), np.log10(sigma_plot),
                           np.log10(r_res_plot))

# Revert logarithm
xi = np.power(10, xi)
yi = np.power(10, yi)
zi = np.power(10, zi)

norm = mc.LogNorm(vmin=zi.min(),
                  vmax=zi.max())
#  max_res = 1e05
#  min_res = 1e-05
#  norm = mc.LogNorm(vmin=min_res,
                  #  vmax=max_res)
#  levels = np.logspace(np.log10(min_res), np.log10(max_res), 11)

# Generate plot
fig = plt.figure()
ax = fig.add_subplot(111)
plt.contour(xi, yi, zi, norm=norm)
plt.contourf(xi, yi, zi, norm=norm)
ax.set_ylabel(r'$\sigma$')
ax.set_xlabel(r'$\rho$')
ax.set_title(r'Ratio between residual norms $\frac{r_p}{r_d}$')
plt.colorbar()
plt.contour(xi, yi, zi, levels=[.1, 10], colors=('k',),linestyles=('-',),linewidths=(2,))
plt.tight_layout()
ax.set_xscale('log')
ax.set_yscale('log')
plt.savefig('sigma_rho_ratio_res.pdf')

'''
Generate plot condition number
'''
# Get grid data
xi, yi, zi = get_grid_data(np.log10(rho_plot), np.log10(sigma_plot),
                           np.log10(cond_KKT_plot))

# Revert logarithm
xi = np.power(10, xi)
yi = np.power(10, yi)
zi = np.power(10, zi)

norm = mc.LogNorm(vmin=zi.min(),
                  vmax=zi.max())

# Generate plot
fig = plt.figure()
ax = fig.add_subplot(111)
plt.contour(xi, yi, zi, norm=norm)
plt.contourf(xi, yi, zi, norm=norm)
ax.set_ylabel(r'$\sigma$')
ax.set_xlabel(r'$\rho$')
ax.set_title(r'Condition number of KKT')
plt.colorbar()
plt.tight_layout()
ax.set_xscale('log')
ax.set_yscale('log')
plt.savefig('sigma_rho_condition_number.pdf')











