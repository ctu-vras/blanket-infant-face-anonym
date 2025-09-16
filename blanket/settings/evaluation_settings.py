from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class EvaluationSettings:
    max_frame_detection_lookback: int = 0
    detection_matching_method: Literal["intersection_over_union", "distance", "confidence"] = "intersection_over_union"
    min_intersection_over_union: float = 0.8
    max_detection_center_distance: int = 200

    evaluate_incorrect_redetections: bool = True

    max_evaluated_frames: Optional[int] = None
    evaluation_skipped_frames: int = 0

    @property
    def sdwebui_server_url(self):
        return f"http://{self.ipv4_address}:{self.port}"
