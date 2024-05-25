import collections
import typing
from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float, Num

from xtils.rl.tabular.types import (
    RewardMatrix,
    RewardVector,
    TabularPolicy,
    TransitionMatrix,
    TransitionTensor,
    ValueVector,
)


@jax.jit
def transition_matrix(P: TransitionTensor, policy: TabularPolicy) -> TransitionMatrix:
    """Compute the transition matrix induced by a given policy."""
    chex.assert_rank([P, policy], {3, 2})
    return jnp.einsum("ast,sa->st", P, policy)


@jax.jit
def reward_vector(R: RewardMatrix, policy: TabularPolicy) -> RewardVector:
    """Compute the reward vector for a given policy."""
    chex.assert_equal_shape([R, policy])
    return jnp.einsum("sa,sa->s", R, policy)


@jax.jit
def successor_representation(P: TransitionTensor, policy: TabularPolicy, discount: Num) -> Float[Array, "s s"]:
    """Compute the successor representation for a given policy."""
    chex.assert_rank([P, policy], {3, 2})
    nstates = P.shape[-1]
    I = jnp.eye(nstates)
    P_d = transition_matrix(P, policy)
    return jnp.linalg.solve(I - discount * P_d, I)


@jax.jit
def uniform_random_policy(P: TransitionTensor) -> TabularPolicy:
    """Uniform random policy."""
    chex.assert_rank(P, 3)
    nactions, nstates = P.shape[0], P.shape[-1]
    return jnp.ones((nstates, nactions)) / nactions


@jax.jit
def proto_value_functions(P: TransitionTensor) -> Float[Array, "s s"]:
    policy = uniform_random_policy(P)
    P_d = transition_matrix(P, policy)
    eigvals, eigvecs = jnp.linalg.eigh(P_d)
    indices = jnp.argsort(eigvals)
    return eigvecs[:, indices]


def bellman_optimality_op(
    P: TransitionTensor,
    R: RewardMatrix,
    discount: Num,
) -> Callable[[ValueVector], ValueVector]:
    """Bellman optimality operator."""
    chex.assert_rank([P, R, discount], {3, 2, 0})

    @jax.jit
    def _operator(v: ValueVector) -> ValueVector:
        return jnp.max(R + discount * jnp.einsum("ast,t->sa", P, v), axis=1)

    return _operator


def policy_improvement_op(
    P: TransitionTensor,
    R: RewardMatrix,
    discount: Num,
) -> Callable[[ValueVector], TabularPolicy]:
    """Policy improvement operator, i.e., the argmax policy w.r.t. the value function."""
    chex.assert_rank([P, R, discount], {3, 2, 0})
    nactions = P.shape[0]

    @jax.jit
    def _operator(v: ValueVector) -> TabularPolicy:
        d = jnp.argmax(R + discount * jnp.einsum("ast,t->sa", P, v), axis=1)  # pyright: ignore
        return jax.nn.one_hot(d, num_classes=nactions)

    return _operator


def policy_evaluation_op(
    P: TransitionTensor,
    R: RewardMatrix,
    discount: Num,
) -> Callable[[TabularPolicy], ValueVector]:
    r"""Policy evaluation operator, i.e., solves (I - \gamma P^\pi)^{-1} r^\pi"""
    chex.assert_rank([P, R, discount], {3, 2, 0})
    nstates = P.shape[-1]
    I = jnp.eye(nstates)

    @jax.jit
    def _operator(pi: TabularPolicy) -> ValueVector:
        P_d = transition_matrix(P, pi)
        R_d = reward_vector(R, pi)
        return jnp.linalg.solve(I - discount * P_d, R_d)

    return _operator


@jax.jit
def value_iteration(
    P: TransitionTensor,
    R: RewardMatrix,
    discount: Num,
    epsilon: float = 1e-12,
) -> Tuple[ValueVector, TabularPolicy]:
    r"""Perform value iteration.

    Solves for the fixed point of v = r + gamma * P * v,
    through fixed point iteration.

    This function uses a crude bound on the l2 norm of the value difference.
    Specifically,
        ||v_{k+1} - v_k||_2 <= \epsilon * (1 - \gamma) / (2 * \gamma)

    """
    chex.assert_rank([P, R, discount, epsilon], {3, 2, 0, 0})
    L = bellman_optimality_op(P, R, discount)

    nstates = P.shape[-1]
    termination = epsilon * (1.0 - discount) / (2 * discount)

    LoopCarry = collections.namedtuple("LoopCarry", ["v", "v_tp1"])

    def _loop_body(carry: LoopCarry) -> LoopCarry:
        v = carry.v_tp1
        v_tp1 = L(v)
        return LoopCarry(v, v_tp1)

    def _loop_cond(carry: LoopCarry) -> bool:
        return jnp.linalg.norm(carry.v_tp1 - carry.v) > termination

    initial_carry = LoopCarry(jnp.zeros(nstates), jnp.ones(nstates))
    final_carry = lax.while_loop(_loop_cond, _loop_body, initial_carry)

    # Final value function
    final_v = final_carry.v_tp1
    # Decision rule, i.e., tabular policy
    final_pi = policy_improvement_op(P, R, discount)(final_v)

    chex.assert_rank([final_v, final_pi], {1, 2})
    return final_v, final_pi


@jax.jit
def policy_iteration(
    P: TransitionTensor,
    R: RewardMatrix,
    discount: Num,
) -> Tuple[ValueVector, TabularPolicy]:
    """Perform policy iteration, i.e., successive policy eval and policy improvement.

    Policy iteration terminates when the policy is stable. That is,
    when the policy no longer changes after the policy improvement step.
    """
    policy_eval = policy_evaluation_op(P, R, discount)
    policy_improvement = policy_improvement_op(P, R, discount)

    nactions, nstates = P.shape[0], P.shape[-1]

    LoopCarry = collections.namedtuple("LoopCarry", ["pi", "pi_tp1"])

    def _loop_body(carry: LoopCarry) -> LoopCarry:
        # Perform policy evaluation.
        pi = carry.pi_tp1
        v = policy_eval(pi)
        # Perform policy improvement.
        pi_tp1 = policy_improvement(v)
        return LoopCarry(pi, pi_tp1)

    def _loop_cond(carry: LoopCarry) -> bool:
        return typing.cast(bool, jnp.logical_not(jnp.all(carry.pi_tp1 == carry.pi)))

    def _random_policy(key: chex.PRNGKey) -> TabularPolicy:
        pi = jax.random.randint(key, (nstates,), minval=0, maxval=nactions - 1)
        return jax.nn.one_hot(pi, num_classes=nactions)

    def _initial_carry() -> LoopCarry:
        key_pi, key_pi_tp1 = jax.random.split(jax.random.PRNGKey(0))
        pi = _random_policy(key_pi)
        pi_tp1 = _random_policy(key_pi_tp1)
        return LoopCarry(pi, pi_tp1)

    initial_carry = _initial_carry()
    final_carry = lax.while_loop(_loop_cond, _loop_body, initial_carry)
    final_pi = final_carry.pi_tp1
    final_v = policy_eval(final_pi)

    chex.assert_rank([final_v, final_pi], {1, 2})
    return final_v, final_pi
