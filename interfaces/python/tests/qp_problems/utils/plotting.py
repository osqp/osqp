import matplotlib as mpl
#  mpl.use('pgf')  # Export pgf figures
import matplotlib.pylab as plt


# Paper stylesheet from: https://gist.github.com/bstellato/e24405efcc532eeda445ea3ab43922f1
#  plt.style.use(['paper'])

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




def create_figure(fig_size):
    if fig_size is not None:
        fig = plt.figure(figsize=figsize(fig_size))
    else:
        fig = plt.figure()

    ax = fig.add_subplot(111)

    return ax
