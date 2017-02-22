from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object
import abc
import utils.data_struct as ds
from future.utils import with_metaclass


class QPExample(with_metaclass(abc.ABCMeta, object)):

    def __init__(self, n_vec, m_vec, rho_vec, sigma_vec, alpha_vec, **kwargs):
        self.create_dims(n_vec, m_vec)
        self.rho_vec = rho_vec
        self.sigma_vec = sigma_vec
        self.alpha_vec = alpha_vec
        self.options = kwargs

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

        # print "\nRun " + self.name() + " tests"
        # print "-----------------------------"

        data_str = ds.data_struct()

        # Cycle over dimensions and rho and sigma and alpha
        # Get total number of problems to be solved
        tot_n_probs = self.dims_mat.shape[1] * len(self.rho_vec) * \
            len(self.sigma_vec) * len(self.alpha_vec)
        counter_prob = 1
        for i in range(self.dims_mat.shape[1]):
            for rho in self.rho_vec:              # iterate over rho values
                for sigma in self.sigma_vec:      # iterate over sigma values
                    for alpha in self.alpha_vec:  # iterate over alpha values
                        # generate problem and store statistics
                        qp = self.gen_problem(self.dims_mat[1, i],
                                              self.dims_mat[0, i],
                                              **self.options)
                        print("Solving %15s: " % self.name() + \
                              "problem %8d of %8d (%.2f %%)" % \
                              (counter_prob, tot_n_probs,
                               counter_prob/tot_n_probs * 100))
                            #   "alpha = %.4e (%d of %d), " % \
                            #   (alpha,
                            #    np.where(self.alpha_vec == alpha)[0][0] + 1,
                            #    len(self.alpha_vec)) + \
                            #   "sigma = %.4e (%d of %d), " % \
                            #   (sigma,
                            #    np.where(self.sigma_vec == sigma)[0][0] + 1,
                            #    len(self.sigma_vec)) + \
                            #   "rho = %.4e (%d of %d)" % \
                            #   (rho, np.where(self.rho_vec == rho)[0][0] + 1,
                            #    len(self.rho_vec))

                        # Solve problem
                        results = qp.solve(rho=rho, sigma=sigma,
                                           alpha=alpha, **kwargs)
                        # Save results
                        data_str.update_data(qp, results, rho, sigma, alpha)

                        # Increment counter
                        counter_prob += 1

            # Dump temporary results file
            data_frame_dump = data_str.get_data_frame()
            data_frame_dump.to_csv('tests/qp_problems/results/%s.csv' % (self.name()), index=False)
            # ipdb.set_trace()

        # Dump final results file
        data_frame_dump = data_str.get_data_frame()
        data_frame_dump.to_csv('tests/qp_problems/results/%s.csv' % (self.name()), index=False)

        # return data frame object
        self.df = data_str.get_data_frame()
