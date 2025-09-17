from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class ModuleSettings:
    """
    Dataclass for module settings (identity encoder, face detector, landmarks detector).
    """

    face_detector_name: str
    facial_landmarks_detector_name: str
    anonymizer_name: str
    identity_encoder_name: str
