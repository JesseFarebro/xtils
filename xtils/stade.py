"""Gym wrapper to convert Gym(nasium) -> dm_env.Environment."""

import functools
import inspect
import typing
from typing import Any, Generator, NamedTuple

import dm_env
import gymnasium as gym
import jax
import more_itertools
import numpy as np
import numpy.random as npr
import wrapt
from dm_env import specs
from gymnasium import spaces


@functools.singledispatch
def flatten_spec(spec) -> tuple[Any, ...]:
    raise ValueError(f"Invalid spec: {type(spec)}.")


@flatten_spec.register
def _(spec: specs.Array) -> tuple[Any, ...]:
    return (spec.shape, spec.dtype), (spec.name,)


@flatten_spec.register
def _(spec: specs.BoundedArray) -> tuple[Any, ...]:
    return (spec.shape, spec.dtype, spec.minimum, spec.maximum), (spec.name,)


@flatten_spec.register
def _(spec: specs.DiscreteArray) -> tuple[Any, ...]:
    return (spec.num_values, spec.dtype), (spec.name,)


@flatten_spec.register
def _(spec: specs.StringArray) -> tuple[Any, ...]:
    return (spec.shape,), (
        spec.string_type,
        spec.name,
    )


jax.tree_util.register_pytree_node(
    specs.Array,
    flatten_spec,
    lambda children, aux_data: specs.Array(*children, *aux_data),
)
jax.tree_util.register_pytree_node(
    specs.BoundedArray,
    flatten_spec,
    lambda children, aux_data: specs.BoundedArray(*children, *aux_data),
)
jax.tree_util.register_pytree_node(
    specs.DiscreteArray,
    flatten_spec,
    lambda children, aux_data: specs.DiscreteArray(*children, *aux_data),
)
jax.tree_util.register_pytree_node(
    specs.StringArray,
    flatten_spec,
    lambda children, aux_data: specs.StringArray(*children, *aux_data),
)


__all__ = ["make", "GymEnvWrapper"]

SpecTuple = tuple[specs.Array, ...]
SpecDict = dict[specs.Array, "SpecDict" | SpecTuple | specs.Array]
SpecTree = specs.Array | SpecDict | SpecTuple | tuple["SpecTree", ...]


@functools.singledispatch
def convert_space(space) -> SpecTree:
    raise ValueError(f"Invalid Gym space: {type(space)}.")


def _(space: spaces.Space[Any]) -> specs.Array:
    """Convert space."""
    return specs.Array(space.shape, space.dtype)


def _(space: spaces.Dict) -> SpecDict:
    """Convert dict space."""
    return jax.tree_util.tree_map(convert_space, space, is_leaf=lambda x: isinstance(x, spaces.Space))


def _(space: spaces.Discrete) -> specs.DiscreteArray:
    """Convert discrete space."""
    return specs.DiscreteArray(space.n, space.dtype)  # pyright: ignore


def _(space: spaces.MultiBinary) -> specs.DiscreteArray:
    """Convert multi-binary space."""
    return specs.DiscreteArray(2, space.dtype)  # pyright: ignore


def _(space: spaces.MultiDiscrete) -> SpecTuple:
    """Convert multi-discrete space."""
    return tuple(
        specs.DiscreteArray(nvalues)
        for nvalues in space.shape  # pyright: ignore
    )


def _(space: spaces.Tuple) -> tuple[SpecTree, ...]:
    """Convert tuple space."""
    return tuple(typing.cast(SpecTree, convert_space(child)) for child in space.spaces)


def _(space: spaces.Box) -> specs.Array:
    """Convert box space."""
    if space.is_bounded("both") or space.is_bounded("above") or space.is_bounded("below"):
        return specs.BoundedArray(space.shape, space.dtype, space.low, space.high)
    else:
        return specs.Array(space.shape, space.dtype)


class GymObservation(NamedTuple):
    """Gym observation tuple."""

    observation: Any
    infos: dict[str, Any]


class GymCompatabilityWrapper(wrapt.ObjectProxy):
    """Gym compatability wrapper for Gym < 0.26."""

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[Any, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if not hasattr(self, "_reset_signature"):
            self._reset_signature = inspect.signature(self.__wrapped__.unwrapped.reset)

        keywords = {}

        if "seed" in self._reset_signature.parameters:
            keywords |= {"seed": seed}
        elif (seed_fn := getattr(self.__wrapped__, "seed", None)) and callable(seed_fn):
            # If the environment doesn't support seed keyword argument it might have
            # a seed function in which case we can just call it before resetting.
            seed_fn(seed)

        if "return_info" in self._reset_signature.parameters:
            keywords |= {"return_info": True}

        if "options" in self._reset_signature.parameters:
            keywords |= {"options": options}

        transition = self.__wrapped__.reset(**keywords)

        # Gym versions < 0.25 return a single observation.
        # Gym versions [0.25, 0.26) can optionally return a 2-tuple with infos.
        # Gym version >= 0.26 returns a 2-tuple with infos.
        match len(transition):
            case 1:
                observation = transition
                infos = {}
            case 2:
                observation, infos = transition
            case _:
                raise ValueError("Invalid number of tuple elements in Gym reset.")

        return observation, infos

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        transition = self.__wrapped__.step(action)

        # Gym >= 0.26 returns a 5-tuple with truncation whereas Gym < 0.26
        # returns a 4-tuple without truncated.
        match len(transition):
            case 4:
                observation, reward, terminal, infos = transition
                truncated = "TimeLimit.truncated" in infos
            case 5:
                observation, reward, terminal, truncated, infos = transition
            case _:
                raise ValueError("Invalid number of tuple elements in Gym transition.")

        return observation, reward, terminal, truncated, infos


class GymEnvWrapper(dm_env.Environment):
    """DeepMind environment wrapper for Gym environments."""

    def __init__(self, env: gym.Env[Any, Any], *, seed: int | jax.Array | npr.SeedSequence | None):
        self._env = GymCompatabilityWrapper(env)
        self._reset_next_step = True

        # Create SeedSequence from PRNGKey or integer.
        match seed:
            case None:
                seed = npr.SeedSequence()
            case jax.Array() | int():
                seed = npr.SeedSequence(np.asarray(seed))

        seed = typing.cast(npr.SeedSequence, seed)
        self._seed = seed.generate_state(n_words=1, dtype=np.uint32).item()

    def reset(self):
        self._reset_next_step = False

        obs, infos = self._env.reset(seed=self._seed)
        # Seed is consumed.
        self._seed = None

        return dm_env.restart(GymObservation(obs, infos))

    def step(self, action: int) -> dm_env.TimeStep:
        if self._reset_next_step:
            return self.reset()

        obs, reward, terminal, truncated, infos = self._env.step(action)

        if terminal:
            timestep = dm_env.termination(reward, GymObservation(obs, infos))
        elif truncated:
            timestep = dm_env.truncation(reward, GymObservation(obs, infos))
        else:
            timestep = dm_env.transition(reward, GymObservation(obs, infos))

        self._reset_next_step = timestep.last()
        return timestep

    @functools.cache
    def action_spec(self) -> SpecTree:
        return convert_space(self._env.action_space)

    @functools.cache
    def observation_spec(self) -> SpecTree:
        return convert_space(self._env.observation_space)

    @functools.cache
    def reward_spec(self) -> SpecTree:
        if reward_range := getattr(self._env, "reward_range", None):
            return specs.BoundedArray((), np.float32, reward_range[0], reward_range[1], name="reward")
        return specs.Array(shape=(), dtype=float, name="reward")

    @more_itertools.consumer
    def __iter__(self) -> Generator[dm_env.TimeStep, int, None]:
        """Infinite environment generator."""
        timestep = self.reset()
        action = yield timestep
        while True:
            timestep = self.step(action)
            action = yield timestep

    def __str__(self) -> str:
        return str(self._env)

    def __repr__(self) -> str:
        return repr(self._env)


def make(
    id: str,
    /,
    *,
    seed: int | jax.Array | npr.SeedSequence | None = None,
    **kwargs,
) -> dm_env.Environment:
    env = gym.make(id, **kwargs)
    return GymEnvWrapper(env, seed=seed)
