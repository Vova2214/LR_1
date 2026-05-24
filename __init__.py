"""Public application-layer service boundary.

Routes should import service singletons from this package boundary. Compatibility
wrappers may import the same exports while older callers are still migrating.
Implementation modules remain internal to the application layer.
"""

from __future__ import annotations

from importlib import import_module

from ..architecture_contracts import APPLICATION_PACKAGE_BOUNDARY

PACKAGE_BOUNDARY = APPLICATION_PACKAGE_BOUNDARY
_EXPORTS = {export.name: (export.module_name, export.attr_name) for export in PACKAGE_BOUNDARY.exports}

__all__ = list(PACKAGE_BOUNDARY.public_exports)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | {"PACKAGE_BOUNDARY"})
