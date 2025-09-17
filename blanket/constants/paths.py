from pathlib import Path
from os import getenv


def override_path(environmental_variable_name: str, default_path: Path, must_exist: bool = True) -> Path:
    """
    Returns a Path object, overridden by environment variable if defined.

    Args:
        environmental_variable_name (str): Name of the environment variable.
        default_path (Path): Default path to use if env_var not defined.
        must_exist (bool): If True, raise error if path doesn't exist.

    Returns:
        Path: Final resolved path.
    """
    path = Path(getenv(environmental_variable_name, default_path)).resolve()

    if must_exist and not path.exists():
        raise RuntimeError(f"Path for {environmental_variable_name} does not exist: {path}")

    return path


# ===== default paths =====
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

_DEFAULT_CONFIG_FOLDER = PROJECT_ROOT / "blanket" / "configs"
_DEFAULT_ANONYMIZER_PARAMETERS_FOLDER = _DEFAULT_CONFIG_FOLDER / "anonymizer_parameters"
_DEFAULT_FACE_DETECTOR_PARAMETERS_FOLDER = _DEFAULT_CONFIG_FOLDER / "detector_parameters" / "face_detector_parameters"
_DEFAULT_FACIAL_LANDMARKS_DETECTOR_PARAMETERS_FOLDER = (
        _DEFAULT_CONFIG_FOLDER / "detector_parameters" / "facial_landmarks_detector_parameters")
_DEFAULT_MODELS_FOLDER = PROJECT_ROOT / "models"
_DEFAULT_SCRIPTS_FOLDER = PROJECT_ROOT / "scripts"
_DEFAULT_LOGS_FOLDER = PROJECT_ROOT / "logs"


# ===== environmental variable overrides =====
CONFIG_FOLDER = override_path("CONFIG_FOLDER", _DEFAULT_CONFIG_FOLDER)
ANONYMIZER_PARAMETERS_FOLDER = override_path("ANONYMIZER_PARAMETERS_FOLDER", _DEFAULT_ANONYMIZER_PARAMETERS_FOLDER)
FACE_DETECTOR_PARAMETERS_FOLDER = override_path(
    "FACE_DETECTOR_PARAMETERS_FOLDER", _DEFAULT_FACE_DETECTOR_PARAMETERS_FOLDER)
FACIAL_LANDMARKS_DETECTOR_PARAMETERS_FOLDER = override_path(
    "FACIAL_LANDMARKS_DETECTOR_PARAMETERS_FOLDER", _DEFAULT_FACIAL_LANDMARKS_DETECTOR_PARAMETERS_FOLDER)
MODELS_FOLDER = override_path("MODELS_FOLDER", _DEFAULT_MODELS_FOLDER)
SCRIPTS_FOLDER = override_path("SCRIPTS_FOLDER", _DEFAULT_SCRIPTS_FOLDER)
LOGS_FOLDER = override_path("LOGS_FOLDER", _DEFAULT_LOGS_FOLDER)
