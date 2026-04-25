"""pytest configuration — add project root to sys.path."""
import sys
from pathlib import Path

# Allow imports from src/ and analysis/
sys.path.insert(0, str(Path(__file__).parent.parent))
