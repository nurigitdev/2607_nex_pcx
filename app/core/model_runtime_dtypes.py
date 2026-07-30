"""Helpers for optional torch dtype runtime configuration and evidence."""

from typing import Any

TORCH_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "torch.bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "torch.float16": "float16",
    "fp32": "float32",
    "float": "float32",
    "float32": "float32",
    "torch.float32": "float32",
}


def normalize_torch_dtype_name(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace("-", "_")
    if not normalized:
        return None
    try:
        return TORCH_DTYPE_ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted({"bfloat16", "float16", "float32"}))
        raise ValueError(
            f"Unsupported torch dtype: {value}. Supported values: {supported}"
        ) from exc


def torch_dtype_from_name(value: str | None) -> object | None:
    dtype_name = normalize_torch_dtype_name(value)
    if dtype_name is None:
        return None
    try:
        import torch
    except ImportError as exc:
        raise ValueError(f"torch is required to use torch dtype {dtype_name}") from exc
    try:
        return getattr(torch, dtype_name)
    except AttributeError as exc:
        raise ValueError(f"torch does not expose dtype {dtype_name}") from exc


def model_first_parameter_dtype_name(model: object) -> str | None:
    parameter = _first_parameter(model)
    dtype = getattr(parameter, "dtype", None)
    if dtype is None:
        return None
    return _runtime_dtype_name(dtype)


def _first_parameter(model: object) -> Any | None:
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(iter(parameters()))
        except StopIteration:
            return None
        except TypeError:
            return None

    nested_model = getattr(model, "model", None)
    if nested_model is not None and nested_model is not model:
        return _first_parameter(nested_model)
    return None


def _runtime_dtype_name(dtype: object) -> str:
    text = str(dtype).strip().lower()
    if text.startswith("torch."):
        text = text.removeprefix("torch.")
    return TORCH_DTYPE_ALIASES.get(text, text)
