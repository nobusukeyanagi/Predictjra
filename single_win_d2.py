#!/usr/bin/env python3
"""Backward-compatible import shim for :mod:`scripts.single_win_d2`.

The canonical D2 implementation lives under ``scripts/``.  This module remains at the
repository root so older local commands/imports keep working without maintaining a
second copy of the model code.
"""
from scripts.single_win_d2 import *  # noqa: F401,F403
