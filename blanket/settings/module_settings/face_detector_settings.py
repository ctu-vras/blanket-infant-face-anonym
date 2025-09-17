from dataclasses import dataclass, field


@dataclass
class FaceDetectorSettings:
    model_name: str = ""
    model_path: str = ""

    minimum_confidence: float = 0.3

    extra_parameters: dict = field(default_factory=dict)
