from builtins import object
import scipy.sparse.linalg as spala
import scipy.sparse as spa
import numpy as np
import numpy.linalg as npla
import pandas as pd
from .qp_problem import QPProblem

SCALING_REG = 1e-8
SCALING_ITER = 15

class data_struct(object):
    """
    Structure for full data collection (with also problem types)
    """
    def __init__(self):
        # Create dictionary of statistics
        self.n_data = []
        self.m_data = []
        self.seed_data = []
        self.name_data = []
        self.rho_data = []
        self.sigma_data = []
        self.alpha_data = []
        self.iter_data = []
        self.time_data = []
        self.pri_res_data = []
        self.dua_res_data = []

        # Additional data to be stored
        # self.condKKT_data = []
        # self.condP_data = []
        # self.froKKT_data = []
        self.trP_data = []
        self.froP_data = []
        self.froA_data = []
        # self.condKKT_bound_data = []
        # self.condP_bound_data = []
        self.norm_q_data = []
        # self.norm_l_data = []
        # self.norm_u_data = []
        self.u_minus_l_data = []

    def scale_qp(self, qp, settings):
        """
        Perform Ruiz equilibration
        """
        scaling_norm = self.work.settings.scaling_norm
        scaling_norm = scaling_norm if scaling_norm == 1 or scaling_norm == 2 \
            else np.inf

        # Get QP variables
        P = qp.P
        q = qp.q
        A = qp.A
        l = qp.l
        u = qp.u

        (m, n) = qp.A.shape

        # Initialize scaling
        d = np.ones(n + m)
        d_temp = np.ones(n + m)

        # Define reduced KKT matrix to scale
        KKT = spa.vstack([
              spa.hstack([P, A.T]),
              spa.hstack([A, spa.csc_matrix((m, m))])]).tocsc()

        # Iterate Scaling
        for i in range(settings.scaling_iter):
            for j in range(n + m):
                if scaling_norm != 2:
                    norm_col_j = spspa.linalg.norm(KKT[:, j],
                                                   scaling_norm)
                else:
                    # Scipy hasn't implemented that function yet!
                    norm_col_j = np.linalg.norm(KKT[:, j].todense(),
                                                scaling_norm)

                if norm_col_j > SCALING_REG:
                    d_temp[j] = 1./(np.sqrt(norm_col_j))

            S_temp = spspa.diags(d_temp)
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

        # Return scaled problem
        return QPProblem(P, q, A, l, u)

    def get_cond_bound(self, M):
        """
        Get bound on condition number of M
        """
        Mb = M
        norm_rows = np.zeros(Mb.shape[0])

        for i in range(len(norm_rows)):
            norm_rows[i] = spa.linalg.norm(Mb[i, :])

        return np.max(norm_rows)/(np.min(norm_rows) + 1e-08)

    def get_distance_lu(self, l, u):
        """
        Get average distance of the noninfinity l and u elements
        """

        # Get index of non infinity elements in l
        l_idx = np.where(np.invert(np.isinf(np.abs(l))))[0]

        # Get index of non infinity elements in u
        u_idx = np.where(np.invert(np.isinf(np.abs(u))))[0]

        # Get intersection of indices
        idx = np.intersect1d(l_idx, u_idx)

        # Get reduced vectors
        l_red = l[idx]
        u_red = u[idx]

        # Get average distance
        avg_dist = np.max(u_red - l_red) if any(idx) else np.inf

        if np.isnan(avg_dist):
            print("ERROR: avg_dist is NaN")

        return avg_dist

    def update_data(self, seed, name, qp, results,
                    rho, sigma, alpha, settings):
        self.n_data.append(qp.n)
        self.m_data.append(qp.m)
        self.rho_data.append(rho)
        self.sigma_data.append(sigma)
        self.alpha_data.append(alpha)
        self.seed_data.append(seed)
        self.name_data.append(name)
        self.iter_data.append(results.info.iter)
        self.time_data.append(results.info.run_time)
        self.pri_res_data.append(results.info.pri_res)
        self.dua_res_data.append(results.info.dua_res)

        # Scale data as OSQP does and store those values
        qp_scaled = self.scale_qp(qp, settings)

        # Compute data statistics
        # KKT = spa.vstack([spa.hstack([qp_scaled.P, qp_scaled.A.T]),
        #                   spa.hstack([qp_scaled.A,
        #                               spa.csc_matrix((qp_scaled.m,
        #                                               qp_scaled.m))])])

        # self.condKKT_data.append(npla.cond(KKT.todense()))
        # self.condP_data.append(npla.cond(qp_scaled.P.todense()))
        # self.froKKT_data.append(spala.norm(KKT))


        self.froP_data.append(spala.norm(qp_scaled.P))
        self.trP_data.append(np.sum(qp_scaled.P.diagonal()))
        self.froA_data.append(spala.norm(qp_scaled.A))
        # self.condKKT_bound_data.append(self.get_cond_bound(KKT.todense()))
        # self.condP_bound_data.append(self.get_cond_bound(qp_scaled.P.todense()))
        self.norm_q_data.append(npla.norm(qp_scaled.q))
        # self.norm_l_data.append(npla.norm(qp_scaled.l))
        # self.norm_u_data.append(npla.norm(qp_scaled.u))
        self.u_minus_l_data.append(self.get_distance_lu(qp_scaled.l,
                                                        qp_scaled.u))

    def get_data_frame(self):
        # Create dictionary
        data = {'n': self.n_data,
                'm': self.m_data,
                'rho': self.rho_data,
                'sigma': self.sigma_data,
                'alpha': self.alpha_data,
                'seed': self.seed_data,
                'name': self.name_data,
                'iter': self.iter_data,
                'time': self.time_data,
                'pri_res': self.pri_res_data,
                'dua_res': self.dua_res_data,
                # 'condKKT': self.condKKT_data,
                # 'condP': self.condP_data,
                # 'froKKT': self.froKKT_data,
                'froP': self.froP_data,
                'trP': self.trP_data,
                'froA': self.froA_data,
                # 'condKKT_bound': self.condKKT_bound_data,
                # 'condP_bound': self.condP_bound_data,
                'norm_q': self.norm_q_data,
                # 'norm_l': self.norm_l_data,
                # 'norm_u': self.norm_u_data,
                'u_minus_l': self.u_minus_l_data
                }

        cols = ['n', 'm', 'rho', 'sigma', 'alpha', 'iter',
                'name', 'time', 'pri_res', 'dua_res', 'seed',
                #  'condKKT',
                # 'condP',
                #  'froKKT',
                'froP', 'trP', 'froA',
                #  'condKKT_bound',
                # 'condP_bound',
                'norm_q',
                # 'norm_l', 'norm_u',
                'u_minus_l']

        # Create dataframe
        df = pd.DataFrame(data)
        df = df[cols]

        return df
