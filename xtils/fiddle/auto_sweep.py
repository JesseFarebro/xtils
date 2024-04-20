"""Fiddle sweep helpers.

This file allows you to write Fiddle "sweeps", i.e., a function which
takes as input a buildable and yields mutated buildables.
You can take the cartesian product of sweeps via `make_trials_from_sweeps`.
You can make an arguments view from a mutated trial via `arguments_from_trial`.

Example:
```py
from typing import Iterator

import fiddle as fdl
from fiddle import selectors

def sweep_lr(base: fdl.Buildable[T]) -> Iterator[fdl.Buildable[T]]:
  for lr in (1e-4, 1e-5, 5e-5):
    yield fdl.deepcopy_with(base, lr=lr)

def sweep_seed(base: fdl.Buildable[T]) -> Iterator[fdl.Buildable[T]]:
  for seed in range(10):
    yield fdl.deepcopy_with(base, seed=seed)


# Trials will contain fdl.Buildables that have (lr, seed)
# pairs mutated from the above configs.
trials = auto_sweep.make_trials_from_sweeps(
    my_base_config(),
    [sweep_lr, sweep_seed]
)

for trial in trials:
  # Trial will be a Buildable with mutated parameters
  args = auto_sweep.arguments_from_trial(trial)
  assert 'seed' in args and 'lr' in args
```

NOTE: Arguments from `arguments_from_trial` shouldn't be passed to XManager.
Instead, you should serialize the trial with `fdl_flags.FiddleFlagSerializer`.
The arguments you obtain from `arguments_from_trial` are purely
for display purposes, e.g., displaying parameters in XManager.
"""

import contextlib
import copy
import dataclasses
import functools
import itertools
import typing
import warnings
from typing import Any, Iterator, Protocol, Sequence, TypeVar

import fiddle as fdl
from fiddle import daglish, diffing, history, printing

T = TypeVar("T")


@typing.runtime_checkable
class SweepFunction(Protocol[T]):
    def __call__(self, base: fdl.Buildable[T]) -> Iterator[fdl.Buildable[T]]: ...


@dataclasses.dataclass(frozen=True)
class _SweepMutationLocation(history.Location):
    """Custom history location that we can query for."""

    ...


@contextlib.contextmanager
def _track_changes() -> Iterator[None]:
    """Track changes which can be later converted into CLI arguments."""

    def _location_provider() -> _SweepMutationLocation:
        # Dummy location provider, we're just using this so we can tell
        # which mutations were made while tracking.
        return _SweepMutationLocation(filename="", line_number=-1, function_name=None)

    is_tracking_enabled = history.tracking_enabled()
    history.set_tracking(True)
    with history.custom_location(_location_provider):
        yield
    history.set_tracking(is_tracking_enabled)


def _make_histories_from_buildable(
    cfg: fdl.Buildable[Any],
) -> Iterator[tuple[daglish.Path, list[history.HistoryEntry]]]:
    """Recursively traverses `cfg` and generates per-param history summaries."""
    for value, path in daglish.iterate(cfg):
        if isinstance(value, fdl.Buildable):
            for name, param_history in value.__argument_history__.items():
                leaf_attr = daglish.BuildableFnOrCls() if name == "__fn_or_cls__" else daglish.BuildableAttr(name)
                subpath = (*path, leaf_attr)
                yield subpath, param_history


def make_trials_from_sweeps(
    base: fdl.Buildable[T],
    sweeps: Sequence[SweepFunction[T]],
    *,
    yield_args: bool = False,
) -> Iterator[fdl.Buildable[T]]:
    """Make trials from sweeps names."""
    if not sweeps:
        yield base
        return

    diffs: list[list[diffing.Diff]] = []
    for sweep_fn in sweeps:
        assert isinstance(sweep_fn, SweepFunction), f"{type(sweep_fn)} is not a SweepFunction"
        diffs.append([diffing.build_diff(base, mutated_cfg) for mutated_cfg in sweep_fn(copy.deepcopy(base))])

    for diffs in itertools.product(*diffs):
        trial = copy.deepcopy(base)
        with _track_changes():
            for diff in diffs:
                diffing.apply_diff(diff, trial)

        # Iterate over mutations and warn the user if there were multiple mutations
        # to the same path for an individual trial.
        for path, histories in _make_histories_from_buildable(trial):
            mutations = list(
                filter(
                    lambda history: isinstance(history.location, _SweepMutationLocation),
                    histories,
                )
            )
            if len(mutations) > 1:
                warnings.warn(
                    f"A sweep resulted in multiple mutations to path {path} for trial "
                    f"{trial!r}. The mutations: "
                    f"{', '.join([repr(mutation.new_value) for mutation in mutations])}"
                )

        yield trial


def arguments_from_trial(base: fdl.Buildable[Any], trial: fdl.Buildable[Any]) -> dict[str, str]:
    """Get arguments from sweep mutations."""
    diff = diffing.build_diff(base, trial)
    skeleton = diffing.skeleton_from_diff(diff)
    diffing.apply_diff(diff, skeleton)
    return printing.as_dict_flattened(skeleton)
