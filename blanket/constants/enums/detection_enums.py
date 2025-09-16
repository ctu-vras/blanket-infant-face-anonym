from enum import Enum


class FaceDetectorModule(str, Enum):
    YOLO = "yolo"
    # ... additional FaceDetectorModules


class FacialLandmarksDetectorModule(str, Enum):
    SPIGA = "spiga"
    # ... additional FacialLandmarksDetectorModules
