import matplotlib as mpl
mpl.use('pgf')  # Export pgf figures
import os
import numpy as np
from .timing import gen_stats_array_vec


def figsize(scale):
    # Get this from LaTeX using \the\textwidth
    fig_width_pt = 469.75
    inches_per_pt = 1.0 / 72.27                       # Convert pt to inch
    # Aesthetic ratio (you could change this)
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0
    fig_width = fig_width_pt * inches_per_pt * scale    # width in inches
    fig_height = fig_width * golden_mean              # height in inches
    # fig_height = fig_width           # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


pgf_with_latex = {                      # setup matplotlib to use latex
    "pgf.texsystem": "pdflatex",        # change this if using xetex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    # blank entries should cause plots to inherit fonts from the document
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 10,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[utf8x]{inputenc}",
        # plots will be generated using this preamble
        r"\usepackage[T1]{fontenc}",
    ]
}
mpl.rcParams.update(pgf_with_latex)
import matplotlib.pylab as plt


def generate_plot(example_name, statistics_name, n_vec, solvers,
                  fig_size=None):

    if fig_size is not None:
        plt.figure(figsize=figsize(fig_size))
    else:
        plt.figure()

    # Plot results
    ax = plt.gca()

    for (solver_name, solver_stats) in solvers.items():
        temp_vec = gen_stats_array_vec(statistics_name, solver_stats)
        plt.semilogy(n_vec, temp_vec, label=solver_name)

    plt.legend()
    plt.grid()
    ax.set_xlabel(r'$n$')
    ax.set_ylabel(r'Time [s]')
    ax.set_title(statistics_name.title())
    plt.tight_layout()
    plt.show(block=False)

    plots_dir = 'scripts/%s/plots' % example_name
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

    # Save figure
    plt.savefig('%s/%s.pdf' % (plots_dir, statistics_name))
