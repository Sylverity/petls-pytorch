"""Lazy access to the optional original PETLS package for parity tests."""

from __future__ import annotations

import importlib


class _LazyPetls:
    def __getattr__(self, name: str):
        return getattr(importlib.import_module("petls"), name)


petls = _LazyPetls()
