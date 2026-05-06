"""
Model registry.

All policy models self-register by decorating their class with
``@register_model("name")``.  The training loop uses ``build_model`` to
instantiate a model purely by its string name, with no hard-coded imports.

Usage
-----
Registering a model (in learning/mymodel/__init__.py)::

    from learning.registry import register_model
    from learning.base import PolicyModel

    @register_model("my_model")
    class MyModel(PolicyModel):
        ...

Building a model by name (in train.py)::

    from learning.registry import build_model
    model = build_model("diffusion_policy", obs_encoder=enc, config=cfg)

Listing available models::

    from learning.registry import list_models
    print(list_models())
"""

from __future__ import annotations

from typing import Any, Type

from learning.base import PolicyModel

_REGISTRY: dict[str, Type[PolicyModel]] = {}


def register_model(name: str):
    """
    Class decorator that registers a PolicyModel subclass under *name*.

    Raises:
        ValueError: if *name* is already registered.
        TypeError:  if the decorated class does not subclass PolicyModel.
    """
    def decorator(cls: Type[PolicyModel]) -> Type[PolicyModel]:
        if name in _REGISTRY:
            raise ValueError(
                f"Model '{name}' is already registered by "
                f"{_REGISTRY[name].__qualname__}."
            )
        if not issubclass(cls, PolicyModel):
            raise TypeError(
                f"{cls.__qualname__} must subclass PolicyModel to be registered."
            )
        _REGISTRY[name] = cls
        return cls

    return decorator


def build_model(name: str, **kwargs: Any) -> PolicyModel:
    """
    Instantiate a registered model by calling its ``build`` classmethod.

    All kwargs are forwarded to ``cls.build(**kwargs)``.  Typically::

        build_model(
            "diffusion_policy",
            obs_encoder=obs_encoder,
            config=dp_config,
        )

    Raises:
        KeyError: if *name* has not been registered.
    """
    if name not in _REGISTRY:
        available = list_models()
        raise KeyError(
            f"Unknown model '{name}'. "
            f"Available models: {available}. "
            f"Make sure the model's package is imported before calling build_model."
        )
    return _REGISTRY[name].build(**kwargs)


def list_models() -> list[str]:
    """Return the names of all registered models, sorted alphabetically."""
    return sorted(_REGISTRY.keys())
