from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from ..constants.enums.anonymization_enums import PaddingMethod


@dataclass
class AnonymizationSettings:
    blacken_without_detections: bool = False
    max_frame_detection_lookback: int = 0
