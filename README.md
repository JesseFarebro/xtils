# E(X)periment Utilities

A set of utilities that I frequently reuse across different projects, 
which would be more efficiently managed in a dedicated repository.

## Jax jit++ (jitpp)

`uv add "xtils[jitpp] @ git+https://github.com/jessefarebro/xtils"`

A wrapper around `jax.jit` providing additional functionality.
This wrapper provides three additional features over regular jit.
  1. Proper type hints for jitted functions. This makes it so tools like
    pyright can show autocompletions for your jitted functions.
  2. You can use type annotations to specify static and donated arguments
    with the `Static[]` and `Donate[]` annotation types.
  3. A somewhat opinionated way to bind attributes of a class so that you
    can jit static class methods more easily while still retaining the
    modularity of classes.

Functional example of type annotations:
```python
from xtils import jitpp
from xtils.jitpp import Static, Donate

@jitpp.jit
def f(x: Donate[int], sign: Static[int]) -> int:
    return x * sign

f(1, -1)
f(1, 1) # re-traced as `sign` is annotated static.
```

> [!CAUTION]
> If you use other decorators between `@jitpp.jit` and your function this could
> potentially cause problems if the type annotations are stripped or if arguments
> are permuted.

Class-based `staticmethod` example:
```python
from xtils import jitpp
from xtils.jitpp import Static, Donate, Bind

@dataclasses.dataclass
class MyClass:
    sign: float

    @jitpp.jit
    @staticmethod
    def f(x: Donate[int], *, sign: Bind[Static[int]]) -> int:
        return x * sign

obj = MyClass(sign=-1)
obj.f(1) # NOTE: sign doesn't need to be provided as its bound to `obj.sign`
obj.sign = 1
obj.f(1) # re-traced as `sign` is annotated static.
```

## Jax debugger++ (jdbpp)

`uv add "xtils[jdbpp] @ git+https://github.com/jessefarebro/xtils"`

An improved version of the builtin Jax debugger. It supports features like:

- An improved UI (e.g., code highlighting, pretty backtrace, prettyprint)
- Ability to run arbitrary commands like pdb, e.g., importing libraries
- IPython shell
- Command history logging so e.g., you can press up to get previous commands from prior sessions

Additional commands include:

- `interact` or `i` to drop into an IPython shell
- `ll` for long-list to list the entire file
- `shape` or `s` to get the shapes of all the variables in the current context
- `pdef`, `pdoc`, `pfile`, `pinfo`, `psource`, and magic commands from IPython
- `v` to get a table of the variables in scope

All you need to do in order to use jdbpp is to import `xtils.jdbpp` and it will register itself with Jax.

## Fiddle

`uv add "xtils[fiddle] @ git+https://github.com/jessefarebro/xtils"`

### Auto Sweep

```py
from typing import Iterator

import fiddle as fdl
from fiddle import selectors

from xtils.fiddle import auto_sweep

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
```

### Printing

```py
from xtils.fiddle import printing

my_config_params = printing.as_dict(
    cfg,
    include_buildable_fn_or_cls=True,
    include_defaults=False,
    buildable_fn_or_cls_key="__fn_or_cls__",
    flatten_tree=False,
)
```

## Common Loop Utils

`uv add "xtils[clu] @ git+https://github.com/jessefarebro/xtils"`

### Metric Writers

```py
from xtils.clu import metric_writers

writer = metric_writers.create_default_writer(
    just_logging=False,
    asynchronous=False,
)
```

You can customize the metric writer with:

```sh
--metric_writer=aim|wandb|tensorboard

# Aim
--aim.repo # Repository directory
--aim.experiment # Experiment name.
--aim.run_hash # Run hash
--aim.log_system_params # Log system parameters.

# Wandb
--wandb.save_code # Save code.
--wandb.id # Run ID
--wandb.tags # Tags.
--wandb.name # Name
--wandb.group # Group.
--wandb.mode # Mode: online|offline|disabled

# Tensorboard
--tensorboard.logdir # Log directory.
```

## Domains

`uv add "xtils[domains] @ git+https://github.com/jessefarebro/xtils"`

### Atari

```py
from xtils.domains import atari as dm_ale

env = dm_ale.AtariEnvironment(
    game,
    mode=None,
    difficulty=None,
    seed=None,
    repeat_action_probability=0.25,
    frameskip=1,
    max_episode_frames=108_000,
    render_mode=None,
    frame_processing=None,
    action_set=ActionSet.Minimal,
    observation_type=(
        dm_ale.ObservationType.ImageRGB,
        dm_ale.ObservationType.Lives,
    )
)
```

## Plotting

`uv add "xtils[plotting] @ git+https://github.com/jessefarebro/xtils"`

### Theme

Get the theme that can be used with Seaborn object `.theme(...)`:

```py
from xtils.plotting import THEME

so.Plot(...)
  .theme(THEME)
```

### Baseline Dataframes

Fetch baseline dataframes for Dopamine and DQN Zoo:

```py
from xtils.plotting import baselines

dopamine = baselines.dopamine()
zoo = baselines.zoo()
```

### Seaborn Objects

- `Rolling` move transform.
- `LineLabel` mark.

#### `Rolling`

Mirrors [`pd.DataFrame.rolling`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html).

```py
from xtils.plotting import objects as xso

so.Plot(...)
  .add(so.Line(), so.Agg(), xso.Rolling())
```

#### `LineLabel`

```py
from xtils.plotting import objects as xso

so.Plot(...)
  .add(so.Line() + xso.LineLabel(), so.Agg(), text="Agent")
```

## RL

`uv add "xtils[rl] @ git+https://github.com/jessefarebro/xtils"`

### Tabular

- Dynamic programming utilities

  - `transition_matrix`
  - `reward_vector`
  - `successor_representation`
  - `uniform_random_policy`
  - `proto_value_functions`
  - `bellman_optimality_op`
  - `policy_improvement_op`
  - `policy_evaluation_op`
  - `value_iteration`
  - `policy_iteration`

- MDP

  - `GridWorld`
  - `FourRoomsGrid`
  - `MiddleWallGrid`
  - `WindingGrid`
  - `DayanGrid`
  - `TMaze`
  - `SimpleT`

- Spectral
  - `policy_mixing_distributions`
  - `compute_entropy`
