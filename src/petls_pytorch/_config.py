"""Validation helpers for object-local device and dtype configuration."""

from typing import cast

import torch

DEFAULT_DEVICE = torch.device("cpu")
DEFAULT_DTYPE = torch.float32


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve an object-local device without changing global configuration.

    ``None`` uses CPU, while ``"auto"`` opts in to CUDA when it is available.
    The default is deliberately CPU so construction never claims a GPU
    unexpectedly.
    """
    if device is None:
        return DEFAULT_DEVICE
    if isinstance(device, str) and device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda:0")
    return resolved


def resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
    """Resolve an object-local floating-point dtype."""
    if dtype is None:
        return DEFAULT_DTYPE
    if isinstance(dtype, str):
        try:
            dtype = cast(torch.dtype, getattr(torch, dtype))
        except AttributeError as exc:
            raise ValueError(f"Unknown torch dtype: {dtype!r}") from exc
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("PETLS supports torch.float32 and torch.float64")
    return dtype
