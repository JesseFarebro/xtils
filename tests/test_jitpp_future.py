from __future__ import annotations
import jax
from xtils import jitpp
from xtils.jitpp import Static


def foo(
    a: jax.Array,
    b: Static[bool],
):
    if b:
        return a + 1
    return a


def test_jitpp_with_future_annotations():
    a = jax.numpy.array([1, 2, 3])

    def check(fn):
        assert fn(a, True).tolist() == [2, 3, 4]
        assert fn(a, False).tolist() == [1, 2, 3]

    # works:
    check(jax.jit(foo, static_argnames=["b"]))

    # with pytest.raises(jax.errors.TracerBoolConversionError):
    check(jitpp.jit(foo))
