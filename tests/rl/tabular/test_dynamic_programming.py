import chex
import jax.numpy as jnp

from xtils.rl.tabular import dynamic_programming as dp
from xtils.rl.tabular import mdp

env = mdp.FourRoomsGrid()
P, R = env.P, env.R
discount = 0.99

policy_eval = dp.policy_evaluation_op(P, R, discount)


def test_value_and_policy_iteration():
    v_1, _ = dp.value_iteration(P, R, discount, epsilon=1e-24)
    v_2, _ = dp.policy_iteration(P, R, discount)
    chex.assert_trees_all_close(v_1, v_2, atol=1e-3)


def test_successor_representation():
    pi = dp.uniform_random_policy(P)
    sr = dp.successor_representation(P, pi, discount)
    chex.assert_shape(sr, (env.num_states, env.num_states))


def test_pvfs():
    pvfs = dp.proto_value_functions(P)
    chex.assert_shape(pvfs, (env.num_states, env.num_states))


def test_sr_pvf_equality():
    pi = dp.uniform_random_policy(P)
    sr = dp.successor_representation(P, pi, discount)
    _, sr_pvfs = jnp.linalg.eigh(sr)
    pvfs = dp.proto_value_functions(P)
    chex.assert_trees_all_close(-sr_pvfs, pvfs, atol=1.5)
