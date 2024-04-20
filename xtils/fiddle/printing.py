import inspect
from typing import Any, Iterator

import fiddle as fdl
from fiddle import daglish
from fiddle._src import printing
from fiddle._src.codegen import formatting_utilities


def as_dict(
    cfg: fdl.Buildable[Any],
    *,
    include_buildable_fn_or_cls: bool = True,
    include_defaults: bool = False,
    buildable_fn_or_cls_key: str = "__fn_or_cls__",
    flatten_tree: bool = False,
) -> dict[str, Any]:
    def _path_to_str(path: daglish.PathElement) -> Any:
        match path:
            case daglish.BuildableFnOrCls():
                return buildable_fn_or_cls_key
            case daglish.Index():
                return path.index
            case daglish.Key():
                return path.key
            case daglish.Attr():
                return path.name
            case _:
                raise ValueError(f"Unexpected path: {path}")

    def dict_generate(value, state=None) -> Iterator[printing._LeafSetting]:
        state = state or daglish.BasicTraversal.begin(dict_generate, value)

        if isinstance(value, fdl.Buildable):
            value = printing._rearrange_buildable_args(value, insert_unset_sentinels=include_defaults)
            if include_buildable_fn_or_cls:
                annotation = printing._get_annotation(cfg, state.current_path)
                yield printing._LeafSetting(
                    state.current_path + (daglish.BuildableFnOrCls(),),
                    annotation,
                    value,
                )

        if not printing._has_nested_builder(value):
            if isinstance(value, printing._UnsetValue):
                value = value.parameter.default
            yield printing._LeafSetting(state.current_path, None, value)
        else:
            assert state.is_traversable(value)
            for sub_result in state.flattened_map_children(value).values:
                yield from sub_result

    args_dict = {}
    for leaf in dict_generate(cfg):
        node = args_dict
        if flatten_tree:
            key = ".".join(map(_path_to_str, leaf.path))
        else:
            for path in leaf.path[:-1]:
                key = _path_to_str(path)
                node.setdefault(key, {})
                node = node[key]
            key = _path_to_str(leaf.path[-1])

        match leaf:
            case printing._LeafSetting(path=(*_, daglish.BuildableFnOrCls())):
                fn_or_cls = fdl.get_callable(leaf.value)
                module = inspect.getmodule(fn_or_cls).__name__
                name = fn_or_cls.__qualname__
                node[key] = f"{module}.{name}"
            case printing._LeafSetting(value=str() | int() | float() | bool() | None):
                node[key] = leaf.value
            case _:
                node[key] = formatting_utilities.pretty_print(leaf.value)

    return args_dict
