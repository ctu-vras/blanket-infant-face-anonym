from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class MainSettings:
    input_folder: str
    output_folder: str
    face_detector_type: str
    face_detector_model_path: str
    face_detector_confidence: float = 0.3
    facial_landmarks_detector_type: str = "spiga"
    facial_landmarks_detector_model_path: str = "models/spiga.pth"
    anonymization_method: str = "black_box"
    log_level: str = "info"

    # Internal defaults
    save_face_detection_visualization: bool = field(default=False)
    save_facial_landmarks_visualization: bool = field(default=False)
    max_frame_detection_lookback: int = field(default=0)
    anonymization_padding_method: str = field(default="ratio")
    anonymization_padding_ratio: float = field(default=0.75)
    anonymization_padding_constant: int = field(default=96)

    @staticmethod
    def from_configs(config_path: Path, defaults_path: Path) -> "MainSettings":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        with open(defaults_path, "r") as f:
            defaults = yaml.safe_load(f)
        # Flatten nested detector configs
        face_detector = config.get("face_detector", {})
        facial_landmarks_detector = config.get("facial_landmarks_detector", {})
        return MainSettings(
            input_folder=config.get("input_folder"),
            output_folder=config.get("output_folder"),
            face_detector_type=face_detector.get("type"),
            face_detector_model_path=face_detector.get("model_path"),
            face_detector_confidence=face_detector.get("confidence", 0.3),
            facial_landmarks_detector_type=facial_landmarks_detector.get("type", "spiga"),
            facial_landmarks_detector_model_path=facial_landmarks_detector.get("model_path", "models/spiga.pth"),
            anonymization_method=config.get("anonymization_method", "black_box"),
            log_level=config.get("log_level", "info"),
            save_face_detection_visualization=defaults.get("save_face_detection_visualization", False),
            save_facial_landmarks_visualization=defaults.get("save_facial_landmarks_visualization", False),
            max_frame_detection_lookback=defaults.get("max_frame_detection_lookback", 0),
            anonymization_padding_method=defaults.get("anonymization_padding_method", "ratio"),
            anonymization_padding_ratio=defaults.get("anonymization_padding_ratio", 0.75),
            anonymization_padding_constant=defaults.get("anonymization_padding_constant", 96),
        )
