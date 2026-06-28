#!/usr/bin/env python3
"""Pytest configuration for robot_retargeter tests."""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test (requires MuJoCo)")


def pytest_collection_modifyitems(config, items):
    """Skip e2e tests unless explicitly requested."""
    if not config.getoption("-m", default=""):
        # If no marker filter specified, skip e2e tests
        skip_e2e = pytest.mark.skip(reason="e2e tests skipped (use -m e2e to run)")
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip_e2e)
