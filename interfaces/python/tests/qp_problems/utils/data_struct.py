from builtins import object
import scipy.sparse.linalg as splinalg
import pandas as pd


class data_struct(object):
    def __init__(self):
        # Create dictionary of statistics
        self.n_data = []
        self.m_data = []
        self.trP_data = []
        self.trA_data = []
        self.froP_data = []
        self.froA_data = []
        self.rho_data = []
        self.sigma_data = []
        self.alpha_data = []
        self.iter_data = []
        self.time_data = []

    def update_data(self, qp, results, rho, sigma, alpha):
        self.n_data.append(qp.n)
        self.m_data.append(qp.m)
        self.trP_data.append(qp.P.diagonal().sum())
        self.trA_data.append(qp.A.diagonal().sum())
        self.froP_data.append(splinalg.norm(qp.P))
        self.froA_data.append(splinalg.norm(qp.A))
        self.rho_data.append(rho)
        self.sigma_data.append(sigma)
        self.alpha_data.append(alpha)
        self.iter_data.append(results.total_iter)
        self.time_data.append(results.cputime)

    def get_data_frame(self):
        # Create dictionary
        data = {'n': self.n_data,
                'm': self.m_data,
                'trP': self.trP_data,
                'trA': self.trA_data,
                'froP': self.froP_data,
                'froA': self.froA_data,
                'rho': self.rho_data,
                'sigma': self.sigma_data,
                'alpha': self.alpha_data,
                'iter': self.iter_data,
                'time': self.time_data}
        cols = ['n', 'm', 'trP', 'trA', 'froP', 'froA', 'rho', 'sigma',
                'alpha', 'iter', 'time']

        # Create dataframe
        df = pd.DataFrame(data)
        df = df[cols]

        return df
