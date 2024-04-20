import functools
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float, Num

from xtils.rl.tabular import dynamic_programming as dp
from xtils.rl.tabular.types import TabularPolicy, TransitionTensor


@functools.partial(jax.jit, static_argnames=("steps",))
def policy_mixing_distributions(
    P: TransitionTensor,
    policy: TabularPolicy,
    *,
    source: int,
    steps: int,
) -> Float[Array, "t s"]:
    """Compute policy mixing distributions for a certain number of steps.

    Note: This function assumes that each "node" in P has out-degree 4. This will
    hold in all the grid-worlds we usually want to visualize.

    Args:
        P: Transition tensor.
        policy: Tabular policy.
        source: What state does mixing begin from?
        steps: How many mixing steps.
    """
    nstates = P.shape[-1]

    # Transition matrix for policy
    M = dp.transition_matrix(P, policy)

    # # Mixing start state
    x = jax.nn.one_hot(source, num_classes=nstates)

    # Runtime O(n^3 * steps)
    def _compute_distribution(
        A: Float[Array, "s s"],
        _,
    ) -> Tuple[Float[Array, "s s"], Float[Array, "s"]]:
        A = A @ M
        distribution = A @ x
        return A, distribution

    _, distributions = lax.scan(_compute_distribution, jnp.eye(nstates), xs=None, length=steps)

    return distributions


@jax.jit
def compute_entropy(distribution: Float[Array, "s"]) -> Num:
    return -distribution @ jnp.where(distribution > 0.0, jnp.log2(distribution), 0.0)
