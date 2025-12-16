import sys
from pathlib import Path

_FACEFUSION_PATH = Path(__file__).parent.parent / "external" / "facefusion"
if _FACEFUSION_PATH.exists() and str(_FACEFUSION_PATH) not in sys.path:
    sys.path.insert(0, str(_FACEFUSION_PATH))
