from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class InputSettings:
    """
    Dataclass for input and output folder settings and related options.
    """

    input_folder: Path
    output_folder: Path

    restrict_access_to_saved: bool = False

    save_comparison_with_original: bool = True

    save_face_detection_visualization: bool = False

    save_facial_landmarks_visualization: bool = False

    save_sdwebui_individual_face_steps: bool = False
    save_sdwebui_individual_face_masks: bool = False
    save_sdwebui_parameters: bool = True

    save_facefusion_parameters: bool = True
