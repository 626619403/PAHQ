"""Root pytest configuration: add src/ to sys.path so packages are importable."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
