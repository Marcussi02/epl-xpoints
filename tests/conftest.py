"""Shared fixtures. Sample CSVs in tests/fixtures use football-data.co.uk's real column
layout; their odds are invented for tests and are not real prices."""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_2425() -> bytes:
    return (FIXTURES / "E0_2425_sample.csv").read_bytes()


@pytest.fixture
def sample_0506() -> bytes:
    return (FIXTURES / "E0_0506_sample.csv").read_bytes()
