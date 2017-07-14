import matplotlib as mpl
# mpl.use('pgf')  # Export pgf figures
import matplotlib.pylab as plt

import os
from .timing import gen_stats_array_vec

# Text width in pt
# -> Get this from LaTeX using \the\textwidth
text_width = 469.75


def figsize(scale):
    fig_width_pt = text_width
    inches_per_pt = 1.0 / 72.27                       # Convert pt to inch
    # Aesthetic ratio (you could change this)
    golden_mean = (5.0 ** 0.5 - 1.0) / 2.0
    fig_width = fig_width_pt * inches_per_pt * scale    # width in inches
    fig_height = fig_width * golden_mean              # height in inches
    # fig_height = fig_width           # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


# Paper stylesheet from:
# https://gist.github.com/bstellato/e24405efcc532eeda445ea3ab43922f1
plt.style.use(['paper'])
#  plt.style.use(['talk'])


def generate_plot(example_name, unit, statistics_name, n_vec, solvers,
                  fig_size=None, plot_name=None):

    if plot_name is None:
        plot_name = example_name

    if fig_size is not None:
        plt.figure(figsize=figsize(fig_size))
    else:
        plt.figure()

    # Plot results
    ax = plt.gca()

    for (solver_name, solver_stats) in solvers.items():
        temp_vec, idx_val = gen_stats_array_vec(statistics_name, solver_stats)
        plt.loglog(n_vec[idx_val], temp_vec, label=r"$\mbox{%s}$" % solver_name)

    plt.legend()
    plt.grid()
    ax.set_xlabel(r'$n$')

    if unit == 'time':
        ax.set_ylabel(r'$\mbox{Time }[s]$')
    elif unit == 'iter':
        ax.set_ylabel(r'$\mathrm{Iterations}$')
    else:
        raise ValueError('Unrecognized y unit')

    #  ax.set_title(statistics_name.title())
    plt.tight_layout()
    # plt.show(block=False)

    plots_dir = 'scripts/%s/plots' % example_name
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

    # Save figure
    plt.savefig('%s/%s_%s.pdf' % (plots_dir, plot_name, statistics_name))
