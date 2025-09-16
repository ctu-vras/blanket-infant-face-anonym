from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from ..constants.enums.anonymization_enums import PaddingMethod


@dataclass
class AnonymizationSettings:
    blacken_without_detections: bool = False
    max_frame_detection_lookback: int = 0

    anonymization_padding_method: PaddingMethod = "ratio"
    anonymization_padding_ratio: float = 0.75
    anonymization_padding_constant: int = 96

    # data validation
    def __post_init__(self):
        """
        Validate types and values after initialization.
        Raises:
            ValueError: If types or values are invalid.
        """
        if type(self.blacken_without_detections) != bool:
            raise ValueError("")

        if self.anonymization_padding_method not in ["ratio", "constant"]:
            raise ValueError("")
