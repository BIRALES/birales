import seaborn as sns
sns.set(color_codes=True)
from itertools import cycle

# Seaborn settings
# This sets reasonable defaults for font size for
# a figure that will go in a paper
sns.set_context("poster")

# Set the font to be serif, rather than sans
sns.set(font='serif')

# Make the background white, and specify the
# specific font family
sns.set_style("ticks",
              {'axes.axisbelow': True,
               'axes.edgecolor': '#666666',
               'axes.facecolor': 'white',
               'axes.grid': True,
               'axes.labelcolor': '.15',
               'axes.spines.bottom': True,
               'axes.spines.left': True,
               'axes.spines.right': True,
               'axes.spines.top': True,
               'figure.facecolor': 'white',
               'font.family': ['serif'],
               'font.sans-serif': ['Arial',
                                   'DejaVu Sans',
                                   'Liberation Sans',
                                   'Bitstream Vera Sans',
                                   'sans-serif'],
               'grid.color': '#e0e0e0',
               'grid.linestyle': '-',
               'image.cmap': 'rocket',
               'lines.solid_capstyle': 'round',
               'lines.linewidth': 5,
               'patch.edgecolor': 'w',
               'patch.force_edgecolor': True,
               'text.color': '.15',
               'xtick.bottom': True,
               'xtick.color': '#666666',
               'xtick.direction': 'out',
               'xtick.top': False,
               'ytick.color': '#666666',
               'ytick.direction': 'out',
               'ytick.left': True,
               'ytick.right': False}
              )

COLORS = [
        "#ba0c3f",
        "#3cb44b",
        '#4b6983',
        "#ffe119",
        "#0082c8",
        "#f58231",
        "#911eb4",
        '#b39169',
        "#46f0f0",
        "#f032e6",
        "#d2f53c",
        "#cb9c9c",
        '#663822',
        "#008080",
        '#eed680',
        "#e6beff",
        '#565248',
        "#aa6e28",
        '#267726',
        "#ff5370",
        "#800000",
        "#aaffc3",
        '#625b81',
        '#c1665a',
        '#314e6c',
        '#d1940c',
        "#808000",
        "#000080",
        '#df421e',
        "#808080",
        "#000000",
        '#807d74',
        '#c5d2c8']

BEAM_COLORS = {}

for i, color in enumerate(COLORS):
    BEAM_COLORS[i] = color

COLORS_C = cycle(COLORS)

C_MAP = sns.dark_palette('#BA0C2F', as_cmap=True)
