from __future__ import annotations

import types

import transformers


def ensure_transformers_deepspeed_attr() -> None:
    """
    Bridge older diffusers EMA code with newer transformers packages.

    Some diffusers versions still call ``transformers.deepspeed`` directly,
    while newer transformers releases only expose DeepSpeed helpers through
    ``transformers.integrations`` or omit them entirely when DeepSpeed is not
    installed. Normal training here does not rely on DeepSpeed, so a stub that
    reports "disabled" is sufficient.
    """
    try:
        transformers.deepspeed  # type: ignore[attr-defined]
        return
    except AttributeError:
        pass

    try:
        from transformers.integrations import deepspeed as ds
    except Exception:
        ds = None

    if ds is None or not hasattr(ds, "is_deepspeed_zero3_enabled"):
        ds = types.SimpleNamespace(is_deepspeed_zero3_enabled=lambda: False)

    transformers.deepspeed = ds  # type: ignore[attr-defined]

    try:
        import diffusers.training_utils as training_utils
    except Exception:
        return

    try:
        training_utils.transformers.deepspeed  # type: ignore[attr-defined]
    except AttributeError:
        training_utils.transformers.deepspeed = ds  # type: ignore[attr-defined]
