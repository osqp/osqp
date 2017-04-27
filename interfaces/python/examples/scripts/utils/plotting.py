import matplotlib.pylab as plt
from .timing import gen_stats_array_vec


def generate_plot(statistics_name, n_vec, solvers):

    #  Plot results
    plt.figure()
    ax = plt.gca()

    for (solver_name, solver_stats) in solvers.items():
        temp_vec = gen_stats_array_vec(statistics_name, solver_stats)
        plt.semilogy(n_vec, temp_vec, label=solver_name)

    plt.legend()
    plt.grid()
    ax.set_xlabel(r'Number of assets $n$')
    ax.set_ylabel(r'Time [s]')
    ax.set_title(statistics_name)
    plt.tight_layout()
    plt.show(block=False)
