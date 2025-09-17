from dataclasses import dataclass, field


@dataclass
class AnonymizerSettings:
    model_name: str = ""
    model_path: str = ""

    requires_face_detections: bool = True
    requires_facial_landmarks_detections: bool = True

    extra_parameters: dict = field(default_factory=dict)
