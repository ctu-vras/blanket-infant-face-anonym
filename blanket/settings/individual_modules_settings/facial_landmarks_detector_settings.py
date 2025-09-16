from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FacialLandmarksDetectorSettings:
    module_name: str

    total_landmarks: int

    detects_orientation: bool = False
    detects_pupils: bool = False

    model_name: str = ""
    model_path: str = ""

    minimum_confidence: float = 0.3

    left_eye_ltrb_landmarks_indices: Optional[np.ndarray] = None
    right_eye_ltrb_landmarks_indices: Optional[np.ndarray] = None
    mouth_ltrb_landmarks_indices: Optional[np.ndarray] = None
    pupils_lr_landmarks_indices: Optional[np.ndarray] = None

    extra_parameters: dict = field(default_factory=dict)
