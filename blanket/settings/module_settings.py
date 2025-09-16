from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from blanket.settings.individual_modules_settings.facefusion_settings import FacefusionSettings
from blanket.settings.individual_modules_settings.sdwebui_settings import SDWebUISettings


@dataclass
class ModuleSettings:
    """
    Dataclass for module settings (identity encoder, face detector, landmarks detector).
    """

    identity_encoder_name: str  # TODO - turn this into an enum
    face_detector_name: str
    facial_landmarks_detector_name: str

    # No __post_init__ needed; this is now a pure data container
