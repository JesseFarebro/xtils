import cycler
import frozendict
import matplotlib as mpl
import seaborn as sns

# Based on science plots retro:
# https://github.com/garrettj403/SciencePlots/blob/master/styles/color/retro.mplstyle
_COLORS = sns.blend_palette([
    "#4165c0",
    "#e770a2",
    "#5ac3be",
    "#696969",
    "#f79a1e",
    "#ba7dcd",
])

_GRID_LINE_WIDTH = 0.8
_STROKE_WIDTH = 2.0
# _GRID_COLOR = "#b0b0b0"
# Open Colors
_GRID_COLOR = "#adb5bd"
_TICK_COLOR = "#868e96"
_LABEL_COLOR = "#868e96"
_TEXT_COLOR = "#343a40"

_TICK_PAD = 5

# Customized Dufte style from mplx
THEME = frozendict.deepfreeze({
    **mpl.rcParams,
    # Color cycle
    "axes.prop_cycle": cycler.cycler(color=_COLORS.as_hex()),  # pyright: ignore
    "font.size": 14,
    "text.color": _TEXT_COLOR,
    "axes.labelcolor": _LABEL_COLOR,
    "axes.labelpad": 18,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "ytick.minor.left": False,
    # Axes aren't used in this theme, but still set some properties in case the user
    # decides to turn them on.
    "axes.edgecolor": _LABEL_COLOR,
    "axes.linewidth": _GRID_LINE_WIDTH * 2,
    # default is "line", i.e., below lines but above patches (bars)
    "axes.axisbelow": True,
    #
    "ytick.right": False,
    "ytick.color": _TICK_COLOR,
    "ytick.major.width": _GRID_LINE_WIDTH * 2,
    "ytick.major.pad": _TICK_PAD,
    "xtick.minor.top": False,
    "xtick.minor.bottom": False,
    "xtick.color": _TICK_COLOR,
    "xtick.major.width": _GRID_LINE_WIDTH * 2,
    "xtick.major.pad": _TICK_PAD,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.labelsize": 18,
    "grid.color": _GRID_COLOR,
    "figure.constrained_layout.h_pad": 0.25,
    "figure.constrained_layout.w_pad": 0.25,
    "lines.linewidth": _STROKE_WIDTH,
    # Choose the line width such that it's very subtle, but still serves as a guide.
    "grid.linewidth": _GRID_LINE_WIDTH,
    "grid.alpha": 0.5,
    "axes.xmargin": 0,
    "axes.ymargin": 0,
    "axes.titlepad": 10,
    "axes.titlesize": 20,
    "text.latex.preamble": r"\usepackage{jmath}",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

__all__ = ["THEME"]
