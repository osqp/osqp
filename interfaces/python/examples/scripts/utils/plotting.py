import matplotlib.pylab as plt
from matplotlib2tikz import save as tikz_save
import os

from .timing import gen_stats_array_vec


def generate_plot(example_name, statistics_name, n_vec, solvers):

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
    ax.set_title(statistics_name.title())
    plt.tight_layout()
    plt.show(block=False)

    plots_dir = 'scripts/%s/plots' % example_name
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

    tikz_save('%s/%s.tex' % (plots_dir, statistics_name),
              figurewidth='.8\\textwidth')
