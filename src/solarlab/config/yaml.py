"""YAML 1.2 scalars and explicit legacy merges; preserve decimal input text."""

from __future__ import annotations

from decimal import Decimal
import re
from typing import Any

import yaml

__all__ = ["load_yaml_mapping"]


class _Loader(yaml.SafeLoader):
    pass


_REMOVED_TAGS = {
    "tag:yaml.org,2002:bool", "tag:yaml.org,2002:int",
    "tag:yaml.org,2002:float", "tag:yaml.org,2002:timestamp",
}
_Loader.yaml_implicit_resolvers = {
    key: [(tag, pattern) for tag, pattern in entries if tag not in _REMOVED_TAGS]
    for key, entries in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_BOOL = re.compile(r"(?:true|True|TRUE|false|False|FALSE)\Z")
_INT = re.compile(r"(?:[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)\Z")
_FLOAT = re.compile(r"(?:[-+]?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?|\.(?:inf|Inf|INF))|\.(?:nan|NaN|NAN))\Z")
_Loader.add_implicit_resolver("tag:yaml.org,2002:bool", _BOOL, list("tTfF"))
_Loader.add_implicit_resolver("tag:yaml.org,2002:int", _INT, list("-+0123456789"))
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    _FLOAT,
    list("-+0123456789."),
)


def _integer(loader: _Loader, node: Any) -> int:
    text = loader.construct_scalar(node)
    if _INT.fullmatch(text) is None:
        raise ValueError("invalid YAML 1.2 integer")
    unsigned = text.lstrip("+-")
    base = 8 if unsigned.startswith("0o") else 16 if unsigned.startswith("0x") else 10
    return int(text, base)


def _decimal(loader: _Loader, node: Any) -> Decimal:
    text = loader.construct_scalar(node)
    if _FLOAT.fullmatch(text) is None:
        raise ValueError("invalid YAML 1.2 float")
    if text.lstrip("+-").lower() in {".inf", ".nan"}:
        raise ValueError("YAML numeric values must be finite")
    value = Decimal(text)
    if not value.is_finite():
        raise ValueError("YAML numeric values must be finite")
    return value


def _boolean(loader: _Loader, node: Any) -> bool:
    text = loader.construct_scalar(node)
    if _BOOL.fullmatch(text) is None:
        raise ValueError("invalid YAML 1.2 boolean")
    return text.lower() == "true"


def _null(loader: _Loader, node: Any) -> None:
    if loader.construct_scalar(node) not in {"", "~", "null", "Null", "NULL"}:
        raise ValueError("invalid YAML 1.2 null")
    return None


def _mapping(loader: _Loader, node: Any) -> dict[str, Any]:
    if not isinstance(node, yaml.MappingNode):
        raise ValueError("configuration mapping tag requires a mapping")
    result: dict[str, Any] = {}
    inherited: dict[str, Any] = {}
    has_merge = False
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            if has_merge:
                raise ValueError("duplicate YAML merge key")
            has_merge = True
            nodes = value_node.value if isinstance(value_node, yaml.SequenceNode) else [value_node]
            for source in nodes:
                if not isinstance(source, yaml.MappingNode):
                    raise ValueError("YAML merge sources must be mappings")
                # Earlier mappings take precedence in a merge sequence;
                # explicit keys below override inherited keys in any order.
                for key, value in loader.construct_object(source, deep=True).items():
                    inherited.setdefault(key, value)
            continue
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str):
            raise ValueError("configuration mapping keys must be strings")
        if key in result:
            raise ValueError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=True)
    return {**inherited, **result}


_Loader.add_constructor("tag:yaml.org,2002:int", _integer)
_Loader.add_constructor("tag:yaml.org,2002:float", _decimal)
_Loader.add_constructor("tag:yaml.org,2002:bool", _boolean)
_Loader.add_constructor("tag:yaml.org,2002:null", _null)
_Loader.add_constructor("tag:yaml.org,2002:map", _mapping)


def _validate_tree(value: Any, ancestors: set[int]) -> None:
    if isinstance(value, (dict, list)):
        if id(value) in ancestors:
            raise ValueError("cyclic YAML aliases are unsupported")
        ancestors.add(id(value))
        for child in value.values() if isinstance(value, dict) else value:
            _validate_tree(child, ancestors)
        ancestors.remove(id(value))
    elif value is not None and type(value) not in {str, int, bool, Decimal}:
        raise ValueError(f"unsupported YAML value type: {type(value).__name__}")


def load_yaml_mapping(content: str | bytes) -> dict[str, Any]:
    """Load one string-keyed document; unsafe tags and cyclic maps are rejected."""
    result = yaml.load(content, Loader=_Loader)
    if not isinstance(result, dict):
        raise ValueError("configuration must be one mapping document")
    _validate_tree(result, set())
    return result
