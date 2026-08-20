#!/usr/bin/env python3
"""Compatibility shim for the active production prediction rules.

New code must import prediction_logic_candidate or prediction_logic_production explicitly.
This module intentionally points to production so legacy imports can never activate an
unapplied candidate during the 13:00 live update.
"""
from prediction_logic_production import *  # noqa: F401,F403
