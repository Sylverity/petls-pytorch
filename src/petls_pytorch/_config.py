"""
Global configuration for PETLS PyTorch backend.

All modules should read from here rather than hard-coding device/dtype.
"""

from typing import Union, cast

import torch

_DEFAULT_DEVICE: torch.device = torch.device("cpu")
_DEFAULT_DTYPE: torch.dtype = torch.float32
_DEFAULT_SPARSE_DTYPE: torch.dtype = torch.float32

# Numerical tolerances for correctness tests against C++ reference
_ATOL: float = 1e-4
_RTOL: float = 1e-3


def get_device() -> torch.device:
    """Return the current default compute device."""
    return _DEFAULT_DEVICE


def get_dtype() -> torch.dtype:
    """Return the current default dtype for dense tensors."""
    return _DEFAULT_DTYPE


def get_sparse_dtype() -> torch.dtype:
    """Return the current default dtype for sparse tensors."""
    return _DEFAULT_SPARSE_DTYPE


def get_tol() -> tuple[float, float]:
    """Return (atol, rtol) for numerical comparisons."""
    return _ATOL, _RTOL


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve an object-local device without changing global configuration.

    ``None`` uses the configured default, while ``"auto"`` opts in to CUDA
    when it is available.  The package default is deliberately CPU so merely
    constructing a PETLS object never claims a GPU unexpectedly.
    """
    if device is None:
        return get_device()
    if isinstance(device, str) and device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda:0")
    return resolved


def resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
    """Resolve an object-local floating-point dtype."""
    if dtype is None:
        return get_dtype()
    if isinstance(dtype, str):
        try:
            dtype = cast(torch.dtype, getattr(torch, dtype))
        except AttributeError as exc:
            raise ValueError(f"Unknown torch dtype: {dtype!r}") from exc
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("PETLS supports torch.float32 and torch.float64")
    return dtype


def set_device(device: Union[str, torch.device]) -> None:
    """Set the default compute device globally."""
    global _DEFAULT_DEVICE
    d = torch.device(device)
    if d.type == "cuda" and d.index is None:
        d = torch.device("cuda:0")
    _DEFAULT_DEVICE = d


def set_dtype(dtype: Union[str, torch.dtype]) -> None:
    """Set the default dtype for dense tensors globally."""
    global _DEFAULT_DTYPE
    if isinstance(dtype, str):
        dtype = cast(torch.dtype, getattr(torch, dtype))
    _DEFAULT_DTYPE = dtype


def set_sparse_dtype(dtype: Union[str, torch.dtype]) -> None:
    """Set the default dtype for sparse tensors globally."""
    global _DEFAULT_SPARSE_DTYPE
    if isinstance(dtype, str):
        dtype = cast(torch.dtype, getattr(torch, dtype))
    _DEFAULT_SPARSE_DTYPE = dtype


def set_tol(atol: float, rtol: float) -> None:
    """Set numerical tolerances for correctness tests."""
    global _ATOL, _RTOL
    _ATOL = atol
    _RTOL = rtol
