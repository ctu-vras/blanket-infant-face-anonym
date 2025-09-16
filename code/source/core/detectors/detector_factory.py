from __future__ import annotations
from pathlib import Path
from typing import TypeVar, Type

from source.settings.config_loader import create_settings_with_extras_from_config_file
from source.settings.individual_modules_settings.face_detector_settings import FaceDetectorSettings
from source.settings.individual_modules_settings.facial_landmarks_detector_settings import (
    FacialLandmarksDetectorSettings)
from source.core.detectors.base_detectors import BaseFaceDetector, BaseFacialLandmarksDetector
from source.constants.enums.detection_enums import FaceDetectorModule, FacialLandmarksDetectorModule

from source.core.detectors.face_detectors.yolo_detector import YOLOFaceDetector
# from source.core.detectors.facial_landmarks_detectors.spiga_detector import SPIGAFacialLandmarksDetector


face_detector_parameters_folder = Path("configs/detector_parameters/face_detector_parameters")

face_detector_registry: dict[FaceDetectorModule, tuple[Type[BaseFaceDetector], Path]] = {
    FaceDetectorModule.YOLO: (YOLOFaceDetector, Path("yolo_parameters.yaml")),
    # ... additional face detectors
}


facial_landmarks_detector_parameters_folder = Path("configs/detector_parameters/facial_landmarks_detector_parameters")

facial_landmarks_detector_registry: dict[FacialLandmarksDetectorModule, tuple[Type[BaseFacialLandmarksDetector], Path]] = {
# FacialLandmarksDetectorModule.SPIGA: (SPIGAFacialLandmarksDetector, Path("spiga_parameters.yaml")),
    # ... additional facial landmarks detectors
}


class DetectorFactory:
    @staticmethod
    def create_face_detector(module: FaceDetectorModule) -> BaseFaceDetector:
        module_registry_entry = face_detector_registry.get(module)

        if module_registry_entry is not None:
            face_detector_class, parameters_filename = module_registry_entry
            detector_parameters = create_settings_with_extras_from_config_file(
                face_detector_parameters_folder / parameters_filename, FaceDetectorSettings)
            return face_detector_class(detector_parameters)
        else:
            raise ValueError(f"Unknown detector module: {module}. "
                             f"Available modules: {list(face_detector_registry.keys())}")

    @staticmethod
    def create_facial_landmarks_detector(module: FacialLandmarksDetectorModule) -> BaseFacialLandmarksDetector:
        module_registry_entry = facial_landmarks_detector_registry.get(module)

        if module_registry_entry is not None:
            facial_landmarks_detector_class, parameters_filename = module_registry_entry
            detector_parameters = create_settings_with_extras_from_config_file(
                facial_landmarks_detector_parameters_folder / parameters_filename, FacialLandmarksDetectorSettings)
            return facial_landmarks_detector_class(detector_parameters)
        else:
            raise ValueError(f"Unknown detector module: {module}. "
                             f"Available modules: {list(facial_landmarks_detector_registry.keys())}")
