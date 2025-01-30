import pytest
import sys
from pathlib import Path

# Add the project root to Python path so tests can find the package
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    ) 