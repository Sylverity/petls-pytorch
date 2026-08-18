"""Focused tests for device-aware profiling."""

from contextlib import nullcontext

import pytest
import torch

import petls_pytorch
from petls_pytorch.core.profile import CudaTimer, Timer


pytestmark = pytest.mark.native


def test_profile_uses_cpu_timer_by_default():
    profile = petls_pytorch.Profile()
    assert isinstance(profile._timer_all, Timer)
    assert profile.device == torch.device("cpu")


def test_complex_profile_follows_complex_device():
    complex_ = petls_pytorch.Complex(boundaries=[], filtrations=[[0.0]], device="cpu")
    assert isinstance(complex_.profile._timer_all, Timer)
    assert complex_.profile.device == torch.device("cpu")


def test_profile_selects_cuda_event_timers(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    profile = petls_pytorch.Profile(device="cuda")
    assert isinstance(profile._timer_all, CudaTimer)
    assert profile._timer_all.device == torch.device("cuda")


def test_profile_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA profiling requested"):
        petls_pytorch.Profile(device="cuda")


def test_cuda_timer_synchronizes_and_reports_seconds(monkeypatch):
    class FakeEvent:
        def __init__(self, enable_timing):
            assert enable_timing

        def record(self):
            pass

        def elapsed_time(self, other):
            return 12.5

    synchronized = []
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: synchronized.append(device))

    timer = CudaTimer("cuda:2")
    timer.start()
    assert timer.stop() == pytest.approx(0.0125)
    assert synchronized == [torch.device("cuda:2")]
