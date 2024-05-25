import dataclasses
import inspect
import typing
from typing import Annotated, Any, Callable, Generic, ParamSpec, TypeGuard, TypeVar

import jax
from jax._src.sharding_impls import UNSPECIFIED, UnspecifiedValue

P = ParamSpec("P")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

__all__ = ["Static", "Donate", "Bind", "jit"]


def _is_annotated_with(typ: Any, sentinel: Any) -> bool:
    """Check if a type is annotated with a specific sentinel."""
    if typing.get_origin(typ) is not Annotated:
        return False
    return sentinel in typing.get_args(typ)


@dataclasses.dataclass(frozen=True)
class Sentinel:
    name: str

    def __repr__(self) -> str:
        return self.name


_StaticSentinel = Sentinel("Static")
_DonateSentinel = Sentinel("Donate")
_BindSentinel = Sentinel("Bind")

Static = Annotated[T, _StaticSentinel]
Donate = Annotated[T, _DonateSentinel]
Bind = Annotated[T, _BindSentinel]


def is_annotated_static(typ: T) -> TypeGuard[Static[T]]:
    """Check if a type is annotated with `Static`."""
    return _is_annotated_with(typ, _StaticSentinel)


def is_annotated_donate(typ: T) -> TypeGuard[Donate[T]]:
    """Check if a type is annotated with `Donate`."""
    return _is_annotated_with(typ, _DonateSentinel)


def is_annotated_bind(typ: T) -> TypeGuard[Bind[T]]:
    """Check if a type is annotated with `Bind`."""
    return _is_annotated_with(typ, _BindSentinel)


class BoundWrapper(jax.stages.Wrapped, Generic[P, T_co]):
    def __init__(self, fn: jax.stages.Wrapped, **kwargs):
        self.fn = fn
        self.kwargs = kwargs

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T_co:
        return self.fn(*args, **kwargs, **self.kwargs)

    def lower(self, *args, **kwargs) -> jax.stages.Lowered:
        return self.fn.lower(*args, **kwargs, **self.kwargs)


class jit(Generic[P, T_co]):
    """A wrapper around `jax.jit` providing additional functionality.

    First, this wrapper provides proper type hints for jitted functions.
    This uses the `ParamSpec` type to allow for type hints on the arguments.

    Additionally, you can also use the following type annotations:
        1. `Static[]` - Mark an argument as static
            (equivalent to `static_argnums` or `static_argnames`)
        2. `Donate[]` - Mark a buffer as donated.
            (equivalent to `donate_argnums`).
            NOTE: Donate can only be used on non __keyword-only__ arguments.
        3. `Bind[]` - When decorating a class `staticmethod` this allows you
            to bind the argument to an instance attribute.
            NOTE: We'll only attempt to bind keyword-only arguments.

    Functional example:
    ```python
    @jit
    def f(x: Donate[int], sign: Static[int]) -> int:
        return x * sign

    f(1, -1)
    f(1, 1) # re-traced as `sign` is annotated static.
    ```

    Class-based `staticmethod` example:
    ```python
    @dataclasses.dataclass
    class MyClass:
        sign: float

        @jit
        @staticmethod
        def f(x: Donate[int], *, sign: Bind[Static[int]]) -> int:
            return x * sign

    obj = MyClass(sign=-1)
    obj.f(1)
    obj.sign = 1
    obj.f(1) # re-traced as `sign` is annotated static.
    ```
    """

    __slots__ = ("fn", "signature", "has_bindings")
    __name__: str = "jit"
    __qualname__: str = "jit"

    def __init__(
        self,
        fn: Callable[P, T_co],
        /,
        *,
        in_shardings: UnspecifiedValue = UNSPECIFIED,
        out_shardings: UnspecifiedValue = UNSPECIFIED,
        keep_unused: bool = False,
        device: jax.Device | None = None,
        backend: str | None = None,
        inline: bool = False,
        abstracted_axes: Any | None = None,
    ) -> None:
        """Initialize the jit decorator."""
        # Unwrap a static function
        if isinstance(fn, staticmethod):
            fn = getattr(fn, "__func__")
        self.signature = inspect.signature(fn)

        # Derive static and donated arguments from the signature
        has_bindings = False
        donate_argnums = set()
        static_argnames = set()
        for index, (name, param) in enumerate(self.signature.parameters.items()):
            if is_annotated_donate(param.annotation):
                assert param.kind is not inspect.Parameter.KEYWORD_ONLY
                donate_argnums.add(index)
            if is_annotated_static(param.annotation):
                static_argnames.add(name)
            if is_annotated_bind(param.annotation) and not has_bindings:
                has_bindings = True
        self.has_bindings = has_bindings

        # Jit the function using
        self.fn = jax.jit(
            fn,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
            keep_unused=keep_unused,
            device=device,
            backend=backend,
            inline=inline,
            abstracted_axes=abstracted_axes,
            donate_argnums=tuple(donate_argnums),
            static_argnames=tuple(static_argnames),
        )

    def __get__(self, obj: T, typ: type[T] | None = None) -> BoundWrapper[P, T_co]:
        """Descriptor method called if `jit` is a method decorator.

        In this case we'll attempt to bind all keyword only arguments annotated with `Bind[]`."""
        del typ

        # Bind any arguments from the instance
        bound = {}
        for name, param in self.signature.parameters.items():
            if not is_annotated_bind(param.annotation):
                continue
            if param.kind is not inspect.Parameter.KEYWORD_ONLY:
                raise ValueError(
                    f"Refusing to bind parameter {name} of kind {param.kind!s} to {self.fn!r}."
                    " Only binding of keyword-only arguments is supported."
                )
            if not hasattr(obj, name):
                raise AttributeError(f"Cannot bind attribute `{name}` from {obj!r} to {self.fn!r}.")
            bound[name] = getattr(obj, name)

        # Bind the arguments with functools.partial
        return BoundWrapper(self.fn, **bound)

    @property
    def __func__(self) -> jax.stages.Wrapped:
        """Allow introspection of the wrapped function."""
        return self.fn

    @property
    def __wrapped__(self) -> jax.stages.Wrapped:
        """Allow introspection of the wrapped function."""
        return self.fn

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T_co:
        """Executes the wrapped function, lowering and compiling as needed."""
        if self.has_bindings:
            ba = filter(
                lambda param: is_annotated_bind(param.annotation),
                self.signature.parameters.values(),
            )
            ba = ", ".join(f"{param.name}: {param.annotation!r}" for param in ba)
            raise ValueError(
                f"Found bound arguments: {ba} on {self.fn!r}, refusing to call the function directly."
                " Bound arguments should only be used on class methods with the `@staticmethod` decorator."
            )
        return self.fn(*args, **kwargs)

    def lower(self, *args: P.args, **kwargs: P.kwargs) -> jax.stages.Lowered:
        """Lower this function explicitly for the given arguments.

        A lowered function is staged out of Python and translated to a
        compiler's input language, possibly in a backend-dependent
        manner. It is ready for compilation but not yet compiled.

        Returns:
        A ``Lowered`` instance representing the lowering."""
        return self.fn.lower(*args, **kwargs)

    def __repr__(self) -> str:
        """Return the representation of the wrapped function."""
        return repr(self.fn)

    @property
    def __isabstractmethod__(self) -> bool:
        """Return whether the wrapped function is abstract."""
        return getattr(self.fn, "__isabstractmethod__", False)
