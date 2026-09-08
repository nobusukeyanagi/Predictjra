#!/usr/bin/env python3
"""Backward-compatible test shim for ``scripts/test_single_win_d2.py``."""
from scripts.test_single_win_d2 import test_feature_contract, test_policy_and_trifecta


if __name__ == "__main__":
    test_feature_contract()
    test_policy_and_trifecta()
    print("OK: single-win D2 synthetic tests passed")
