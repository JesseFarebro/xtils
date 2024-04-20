# E(X)periment Utilities

## Fiddle

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

- `GaussianSmooth` move transform.
- `LineLabel` mark.

#### `GaussianSmooth`

```py
from xtils.plotting import objects as xso

so.Plot(...)
  .add(so.Line(), so.Agg(), xso.GaussianSmooth(sigma=1))
```

#### `LineLabel`

```py
from xtils.plotting import objects as xso

so.Plot(...)
  .add(so.Line() + xso.LineLabel(), so.Agg(), text="Agent")
```

## RL

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
