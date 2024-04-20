from jaxtyping import Array, Float

TransitionTensor = Float[Array, "a s s"]
TransitionMatrix = Float[Array, "s s"]
RewardMatrix = Float[Array, "s a"]
RewardVector = Float[Array, "s"]
ValueVector = Float[Array, "s"]
TabularPolicy = Float[Array, "s a"]
