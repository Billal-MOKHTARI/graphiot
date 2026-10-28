import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def tag_map_path(root) -> Path:
    return root / "plant" / "tag_map.csv"


@pytest.fixture(scope="session")
def tags(tag_map_path):
    from graphiot.plant.tags import load_tag_map
    return load_tag_map(tag_map_path)


def pytest_configure():
    # Keep runs reproducible and off the real network.
    os.environ.setdefault("GRAPHIOT_TESTING", "1")
