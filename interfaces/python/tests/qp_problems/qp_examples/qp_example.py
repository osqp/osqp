from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import sys   # To get maxsize

# Metadata class
import abc
from future.utils import with_metaclass

# Data structures
from utils.data_struct import data_struct, full_data_struct

class QPExample(with_metaclass(abc.ABCMeta, object)):

    def __init__(self, n_vec, m_vec, rho_vec, sigma_vec, alpha_vec, nm_num_prob, **kwargs):
        self.dims_mat = self.create_dims(n_vec, m_vec)
        self.rho_vec = rho_vec
        self.sigma_vec = sigma_vec
        self.alpha_vec = alpha_vec
        self.nm_num_prob = nm_num_prob
        self.options = kwargs
        self.df = None
        self.full_df = None

    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def create_dims(self, n_vec, m_vec):
        """Create problem dimensions making it feasible
        """
        pass

    @abc.abstractmethod
    def gen_problem(self, n_vec, m_vec, **kwargs):
        """Generate QP problem
        """
        pass

    def perform_tests(self, **kwargs):

        # Create data structures for storing
        data_str = data_struct()
        full_data_str = full_data_struct()

        # Cycle over dimensions and rho and sigma and alpha
        # Get total number of problems to be solved
        tot_n_probs = self.dims_mat.shape[1] * self.nm_num_prob * len(self.rho_vec) * \
            len(self.sigma_vec) * len(self.alpha_vec)

        # Initialize big for concatenation
        counter_prob = 1
        for i in range(self.dims_mat.shape[1]):
            for _ in range(self.nm_num_prob):  # Generate some random problems

                # Get current seed
                current_seed = np.random.randint(0, 2**32 - 1)
                np.random.seed(current_seed)

                # generate problem and store statistics
                qp = self.gen_problem(self.dims_mat[1, i],  # m
                                      self.dims_mat[0, i],  # n
                                      **self.options)

                for rho in self.rho_vec:              # iterate over rho values
                    for sigma in self.sigma_vec:      # iterate over sigma values
                        for alpha in self.alpha_vec:  # iterate over alpha values
                            print("Solving %15s: " % self.name() + \
                                  "problem %8d of %8d (%.2f %%)" % \
                                  (counter_prob, tot_n_probs,
                                   counter_prob/tot_n_probs * 100))


                            # Solve problem
                            results = qp.solve(rho=rho, sigma=sigma,
                                               alpha=alpha, **kwargs)

                            # Save results into standard data struct
                            data_str.update_data(current_seed, self.name(),
                                                 qp, results,
                                                 rho, sigma, alpha)

                            # Save results into full data struct
                            full_data_str.update_data(current_seed,
                                                      self.name(), qp, results,
                                                      rho, sigma, alpha, kwargs)

                            # Increment counter
                            counter_prob += 1

                            #  # Dump temporary results file
                            #  data_frame_dump = data_str.get_data_frame()
                            #  data_frame_dump.to_csv('results/%s.csv' %
                                                   #  (self.name()), index=False)
                            #  full_data_frame_dump = \
                                #  full_data_str.get_data_frame()
                            #  full_data_frame_dump.to_csv('results/%s_full.csv' %
                                                   #  (self.name()), index=False)

                            # ipdb.set_trace()

            # Dump final results file
            data_frame_dump = data_str.get_data_frame()
            data_frame_dump.to_csv('results/%s.csv' % (self.name()), index=False)

            full_data_frame_dump = full_data_str.get_data_frame()
            full_data_frame_dump.to_csv('results/%s_full.csv' % (self.name()), index=False)

        # return data frame object
        self.df = data_str.get_data_frame()
        self.full_df = full_data_str.get_data_frame()
