"""Create grid world environments from string layouts"""

import abc
import functools
import itertools

import numpy as np
import scipy.ndimage
from jaxtyping import Array, Float, Num
from matplotlib import collections, colors, patches
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d.axes3d import Axes3D

TransitionTensor = Float[Array, "a s s"]
RewardMatrix = Float[Array, "s a"]


class MDP(abc.ABC):
    @property
    @abc.abstractmethod
    def P(self) -> TransitionTensor:
        """Transition tensor"""
        ...

    @property
    @abc.abstractmethod
    def R(self) -> RewardMatrix:
        """Reward matrix"""
        ...

    @property
    @abc.abstractmethod
    def num_states(self) -> Num:
        """Number of states"""
        ...

    @property
    @abc.abstractmethod
    def num_actions(self) -> Num:
        """Number of actions"""
        ...


# Adapted from https://github.com/pierrelux/rooms
class GridWorld(MDP):
    def __init__(self, layout: str, subsample: int = 2) -> None:
        self._layout = layout
        self._grid = np.array([[0 if char == "w" else 1 for char in line] for line in self._layout.splitlines()])
        self._grid = scipy.ndimage.zoom(self._grid, subsample, order=0)

        directions = [
            np.array((-1, 0)),  # UP
            np.array((1, 0)),  # DOWN
            np.array((0, -1)),  # LEFT
            np.array((0, 1)),
        ]  # RIGHT

        self._state_to_grid_cell = np.argwhere(self._grid)
        self._grid_cell_to_state = {
            tuple(self._state_to_grid_cell[s].tolist()): s for s in range(self._state_to_grid_cell.shape[0])
        }

        nstates = self._state_to_grid_cell.shape[0]
        nactions = len(directions)
        self._P = np.zeros((nactions, nstates, nstates))
        for state, idx in enumerate(self._state_to_grid_cell):
            for action, d in enumerate(directions):
                if self._grid[tuple(idx + d)]:
                    dest_state = self._grid_cell_to_state[tuple(idx + d)]
                    self._P[action, state, dest_state] = 1.0
                else:
                    self._P[action, state, state] = 1.0

        self._R = np.copy(np.swapaxes(self.P[:, :, -1], 0, 1))

    @property
    def P(self) -> TransitionTensor:
        return self._P  # pyright: ignore

    @property
    def R(self) -> RewardMatrix:
        return self._R  # pyright: ignore

    @property
    def num_actions(self) -> Num:
        return 4

    @property
    def num_states(self) -> Num:
        return self.P.shape[-1]

    @functools.singledispatchmethod
    def render(self, axis: Axes | Axes3D, values: Float[Array, "s"], /, **kwargs) -> None:
        pass

    @render.register
    def _(self, axis: Axes3D, values: Float[Array, "s"], /, **kwargs) -> None:
        # Set axis limits
        nrows, ncols = self._grid.shape
        # Customize axis.
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(0, ncols)
        axis.set_ylim(nrows, 0)
        axis.grid(False)

        axis.spines.right.set_visible(False)
        axis.spines.top.set_visible(False)
        axis.spines.bottom.set_visible(False)
        axis.spines.left.set_visible(False)

        axis.xaxis.pane.fill = False
        axis.yaxis.pane.fill = False
        axis.zaxis.pane.fill = False
        axis.xaxis.pane.set_edgecolor(None)
        axis.yaxis.pane.set_edgecolor(None)
        axis.zaxis.pane.set_edgecolor(None)
        axis.set_axis_off()

        axis.tick_params(left=False, bottom=False)
        axis.set(xticklabels=[], yticklabels=[])

        # Collect x, y patches
        xs, ys = np.meshgrid(np.arange(0, self._grid.shape[1]), np.arange(0, self._grid.shape[0]))
        zs = np.zeros_like(self._grid).tolist()
        for row, col in itertools.product(range(nrows), range(ncols)):
            if self._grid[row, col] == 0:
                if (
                    (row + 1 < nrows and col + 1 < ncols and self._grid[row + 1, col + 1] != 0)
                    or (col + 1 < ncols and self._grid[row, col + 1] != 0)
                    or (row + 1 < nrows and self._grid[row + 1, col] != 0)
                    or (row - 1 >= 0 and col - 1 >= 0 and self._grid[row - 1, col - 1] != 0)
                    or (col - 1 >= 0 and self._grid[row, col - 1] != 0)
                    or (row - 1 >= 0 and self._grid[row - 1, col] != 0)
                    or (row + 1 < nrows and col - 1 >= 0 and self._grid[row + 1, col - 1] != 0)
                    or (row - 1 >= 0 and col + 1 < ncols and self._grid[row - 1, col + 1] != 0)
                ):
                    rectangle = patches.Rectangle((col, row), 2, 2, facecolor="#343a40", edgecolor="none")
                    axis.add_patch(rectangle)
                    art3d.pathpatch_2d_to_3d(rectangle, z=0, zdir="z")
            else:
                zs[row][col] = values[self._grid_cell_to_state[row, col]].item()

        zs = np.asarray(zs)
        zs_z = np.ones_like(zs) * 0.1

        cmap = kwargs.get("cmap")
        # zs = colors.Normalize()(zs)
        zs = colors.LogNorm()(zs)
        facecolors = cmap(zs)
        print(facecolors)
        # facecolors[np.where(self._grid == 0)] = (52 / 255, 58 / 255, 64 / 255, 1.0)
        facecolors[np.where(self._grid == 0)] = (1, 1, 1, 1.0)

        edgecolor = np.zeros_like(facecolors)
        edgecolor[np.where(self._grid != 0)] = (233 / 255, 236 / 255, 239 / 255, 1.0)

        # p = patches.Circle(
        #     (8.5, 4.5),
        #     1,
        #     facecolor="#fde725",
        #     zorder=5,
        #     alpha=0.85,
        # )
        # axis.add_patch(p)
        # art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

        # p = patches.Circle(
        #     (8.5, 4.5),
        #     1.15,
        #     facecolor="none",
        #     edgecolor="#fde725",
        #     lw=2.5,
        #     zorder=6,
        #     alpha=0.95,
        # )
        # axis.add_patch(p)
        # art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

        # p = patches.Circle(
        #     (2, 2),
        #     0.5,
        #     facecolor="#fde725",
        #     zorder=5,
        #     alpha=0.85,
        # )
        # axis.add_patch(p)
        # art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

        # p = patches.Circle(
        #     (2, 2),
        #     0.65,
        #     facecolor="none",
        #     edgecolor="#fde725",
        #     lw=2.5,
        #     zorder=6,
        #     alpha=0.95,
        # )
        # axis.add_patch(p)
        # art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

        axis.plot_surface(
            xs,
            ys,
            zs_z,
            linewidth=0,
            facecolors=facecolors,
            # edgecolor="none",
            # edgecolor=edgecolor,
            # edgecolors=edgecolor,
            zorder=2,
            # alpha=0.65,
            alpha=1.0,
            **kwargs,
        )

    @render.register
    def _(self, axis: Axes, values: Float[Array, "s"], /, **kwargs) -> None:
        # Set axis limits
        nrows, ncols = self._grid.shape
        # Customize axis.
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(0, ncols)
        axis.set_ylim(nrows, 0)
        axis.spines.right.set_visible(False)
        axis.spines.top.set_visible(False)
        axis.spines.bottom.set_visible(False)
        axis.spines.left.set_visible(False)
        axis.tick_params(left=False, bottom=False)
        axis.set(xticklabels=[], yticklabels=[])

        # Collect x, y patches
        wall_patches, value_patches = [], []
        value_patches = []
        for row, col in itertools.product(range(nrows), range(ncols)):
            rectangle = patches.Rectangle((col, row), 1, 1)
            if self._grid[row, col] == 0:
                wall_patches.append(rectangle)
            else:
                value_patches.append(rectangle)

        # Add wall patches
        wall_collection = collections.PatchCollection(
            wall_patches,
            facecolor="#343a40",
            edgecolor=None,
            linewidth=0.0,
        )
        axis.add_collection(wall_collection)  # pyright: ignore

        # Add value patches
        value_collection = collections.PatchCollection(value_patches, edgecolor=None, linewidth=0.0, **kwargs)
        value_collection.set_array(values)
        axis.add_collection(value_collection)  # pyright: ignore


FourRoomsGrid = functools.partial(
    GridWorld,
    layout="""\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww\
""",
)

MiddleWallGrid = functools.partial(
    GridWorld,
    layout="""\
wwwwwwwwwwwww
w     w     w
w     w     w
w     w     w
w     w     w
w     w     w
w           w
w     w     w
w     w     w
w     w     w
w     w     w
w     w     w
wwwwwwwwwwwww\
""",
)

WindingGrid = functools.partial(
    GridWorld,
    layout="""\
wwwwwwwwwwwww
w           w
wwwwwwwwwww w
w           w
w wwwwwwwwwww
w           w
wwwwwwwwwww w
w           w
w wwwwwwwwwww
w           w
wwwwwwwwwww w
w           w
wwwwwwwwwwwww\
""",
)

DayanGrid = functools.partial(
    GridWorld,
    layout="""\
wwwwwwwwwwwww
w           w
w     wwww  w
w     w     w
w     w     w
w     w     w
w     wwww  w
w           w
wwwwwwwwwwwww\
""",
)


TMaze = functools.partial(
    GridWorld,
    subsample=5,
    layout="""\
wwwwwwwwwww
w         w
wwwww wwwww
wwwww wwwww
wwwww wwwww
wwwww wwwww
wwwww wwwww
wwwww wwwww
wwwww wwwww
wwwww wwwww
wwwww wwwww
wwwwwwwwwww
""",
)

SimpleT = functools.partial(
    GridWorld,
    subsample=1,
    layout="""\
wwwww
w   w
ww ww
wwwww\
""",
)
