from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from code.source.core.detectors.base_detectors import BaseFaceDetector, BaseFacialLandmarksDetector
from code.source.constants.enums.detection_enums import FaceDetectorModule, FacialLandmarksDetectorModule
from code.source.core.detectors.detector_factory import DetectorFactory

from code.source.settings.individual_modules_settings.sdwebui_settings import SDWebUISettings
from code.source.settings.individual_modules_settings.facefusion_settings import FacefusionSettings


@dataclass
class ModuleSettings:
    identity_encoder_name: str  # TODO - turn this into an enum
    face_detector_name: str
    facial_landmarks_detector_name: str

    face_detector: BaseFaceDetector = field(init=False)
    facial_landmarks_detector: BaseFacialLandmarksDetector = field(init=False)

    def __post_init__(self):
        # converting module names from strings to enums
        try:
            face_detector_module = FaceDetectorModule(self.face_detector_name)
        except ValueError:
            raise ValueError(
                f"Unknown face detector '{self.face_detector_name}'. "
                f"Available options: {[module.value for module in FaceDetectorModule]}"
            )

        try:
            facial_landmarks_detector_module = FacialLandmarksDetectorModule(self.facial_landmarks_detector_name)
        except ValueError:
            raise ValueError(
                f"Unknown facial landmarks detector '{self.facial_landmarks_detector_name}'. "
                f"Available options: {[module.value for module in FacialLandmarksDetectorModule]}"
            )

        # initialize detectors through factory
        self.face_detector = DetectorFactory.create_face_detector(face_detector_module)
        self.facial_landmarks_detector = DetectorFactory.create_facial_landmarks_detector(facial_landmarks_detector_module)
