from __future__ import annotations
from pathlib import Path
from typing import Type, Optional

from blanket.constants.paths import FACE_DETECTOR_PARAMETERS_FOLDER, FACIAL_LANDMARKS_DETECTOR_PARAMETERS_FOLDER
from blanket.constants.enums.detection_enums import FaceDetectorModule, FacialLandmarksDetectorModule
from blanket.core.detectors.base_detectors import BaseFaceDetector, BaseFacialLandmarksDetector
from blanket.core.config_loader import create_settings_with_extras_from_config_file
from blanket.settings.module_settings.face_detector_settings import FaceDetectorSettings
from blanket.settings.module_settings.facial_landmarks_detector_settings import (
    FacialLandmarksDetectorSettings,
)

from blanket.core.detectors.face_detectors.yolo_detector import YOLOFaceDetector

from blanket.core.detectors.facial_landmarks_detectors.spiga_detector import SPIGAFacialLandmarksDetector


face_detector_registry: dict[FaceDetectorModule, tuple[Type[BaseFaceDetector], Path]] = {
    FaceDetectorModule.YOLO: (YOLOFaceDetector, Path("yolo_parameters.yaml")),
    # ... additional face detectors
}


facial_landmarks_detector_registry: dict[
    FacialLandmarksDetectorModule, tuple[Type[BaseFacialLandmarksDetector], Path]
] = {
    FacialLandmarksDetectorModule.SPIGA: (SPIGAFacialLandmarksDetector, Path("spiga_parameters.yaml")),
    # ... additional facial landmarks detectors
}


class DetectorFactory:
    @staticmethod
    def create_face_detector(module: FaceDetectorModule) -> BaseFaceDetector:
        """
        Create a face detector instance for the given module.
        Args:
            module (FaceDetectorModule): Enum value for face detector type.
        Returns:
            BaseFaceDetector: Instantiated face detector.
        Raises:
            ValueError: If module is unknown.
        """
        module_registry_entry = face_detector_registry.get(module)

        if module_registry_entry is not None:
            face_detector_class, parameters_filename = module_registry_entry
            detector_parameters = create_settings_with_extras_from_config_file(
                FACE_DETECTOR_PARAMETERS_FOLDER / parameters_filename, FaceDetectorSettings
            )
            return face_detector_class(detector_parameters)
        else:
            raise ValueError(
                f"Unknown detector module: {module}. " f"Available modules: {list(face_detector_registry.keys())}"
            )

    @staticmethod
    def create_facial_landmarks_detector(module: FacialLandmarksDetectorModule) -> BaseFacialLandmarksDetector:
        """
        Create a facial landmarks detector instance for the given module.
        Args:
            module (FacialLandmarksDetectorModule): Enum value for facial landmarks detector type.
        Returns:
            BaseFacialLandmarksDetector: Instantiated facial landmark's detector.
        Raises:
            ValueError: If module is unknown.
        """
        module_registry_entry = facial_landmarks_detector_registry.get(module)

        if module_registry_entry is not None:
            facial_landmarks_detector_class, parameters_filename = module_registry_entry
            detector_parameters = create_settings_with_extras_from_config_file(
                FACIAL_LANDMARKS_DETECTOR_PARAMETERS_FOLDER / parameters_filename, FacialLandmarksDetectorSettings
            )
            return facial_landmarks_detector_class(detector_parameters)
        else:
            raise ValueError(
                f"Unknown detector module: {module}. "
                f"Available modules: {list(facial_landmarks_detector_registry.keys())}"
            )


# TODO - consider creating one large cache in cache.py that would store detectors, anonymizers,...
class DetectorCache:
    """
    Use:
        detector = DetectorCache.get_face_detector(FaceDetectorModule.YOLO)
        detector = DetectorCache.get_facial_landmarks_detector(FaceDetectorModule.SPIGA)
    """
    _cache_instance: Optional[DetectorCache] = None

    def __init__(self):
        if DetectorCache._cache_instance is not None:
            raise RuntimeError("Trying to initialize DetectorCache multiple times")

        self._cache: dict[str, BaseFaceDetector | BaseFacialLandmarksDetector] = {}

    @classmethod
    def get_face_detector(cls, module: FaceDetectorModule) -> BaseFaceDetector:
        return cls._get_cache_instance()._get_face_detector(module)

    def _get_face_detector(self, module: FaceDetectorModule) -> BaseFaceDetector:
        key = f"face:{module.value}"
        if key not in self._cache:
            self._cache[key] = DetectorFactory.create_face_detector(module)
        return self._cache[key]

    @classmethod
    def get_facial_landmarks_detector(cls, module: FacialLandmarksDetectorModule) -> BaseFacialLandmarksDetector:
        return cls._get_cache_instance()._get_facial_landmarks_detector(module)

    def _get_facial_landmarks_detector(self, module: FacialLandmarksDetectorModule) -> BaseFacialLandmarksDetector:
        key = f"landmarks:{module.value}"
        if key not in self._cache:
            self._cache[key] = DetectorFactory.create_facial_landmarks_detector(module)
        return self._cache[key]

    def clear(self):
        """Optional: clear the cache if you need to reload models."""
        self._cache.clear()

    @classmethod
    def _get_cache_instance(cls):
        if cls._cache_instance is None:
            cls._cache_instance = DetectorCache()
        return cls._cache_instance
