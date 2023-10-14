import sys
from pathlib import Path

PACKAGE_PATH = str(Path(__file__).parent.parent)
sys.path.insert(0, PACKAGE_PATH)

pytest_plugins = ["tests.fixtures"]
