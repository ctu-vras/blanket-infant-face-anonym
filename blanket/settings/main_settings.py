from __future__ import annotations
from typing import ClassVar, Optional
from dataclasses import fields
from pathlib import Path

from blanket.core.config_loader import config_file_to_dictionary
from blanket.settings.input_settings import InputSettings
from blanket.settings.module_settings import ModuleSettings
from blanket.settings.anonymization_settings import AnonymizationSettings
from blanket.settings.evaluation_settings import EvaluationSettings
from blanket.settings.logging_settings import LoggingSettings


class MainSettings:
    _settings_instance: ClassVar[Optional[MainSettings]] = None
    _config_filepath: ClassVar[Optional[Path]] = None

    def __init__(self, config_filepath: Path):
        if MainSettings._settings_instance is not None:
            raise RuntimeError("Trying to initialize MainSettings multiple times")

        config_dict = config_file_to_dictionary(config_filepath)

        self.input_settings: InputSettings = InputSettings(**self._filter_invalid_fieldnames(config_dict.get("input_settings", {}), InputSettings))
        self.module_settings: ModuleSettings = ModuleSettings(**self._filter_invalid_fieldnames(config_dict.get("module_settings", {}), ModuleSettings))
        self.anonymization_settings: AnonymizationSettings = AnonymizationSettings(**self._filter_invalid_fieldnames(config_dict.get("anonymization_settings", {}), AnonymizationSettings))
        self.evaluation_settings: EvaluationSettings = EvaluationSettings(**self._filter_invalid_fieldnames(config_dict.get("evaluation_settings", {}), EvaluationSettings))
        self.logging_settings: LoggingSettings = LoggingSettings(**self._filter_invalid_fieldnames(config_dict.get("logging_settings", {}), LoggingSettings))

    @staticmethod
    def _filter_invalid_fieldnames(config_dict: dict, settings_class):
        valid_keys = {f.name for f in fields(settings_class)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return filtered_dict

    @classmethod
    def configure(cls, config_filepath: Path) -> None:
        cls._config_filepath = config_filepath

    @classmethod
    def get(cls) -> MainSettings:
        if cls._config_filepath is None:
            raise RuntimeError("Trying to access MainSettings before it has been configured")
        if cls._settings_instance is None:
            cls._settings_instance = MainSettings(cls._config_filepath)
        return cls._settings_instance





#
# class MainSettings:
#     # Actual settings
#     input_settings: "InputSettings"
#     module_settings: "ModuleSettings"
#     anonymization_settings: "AnonymizationSettings"
#     evaluation_settings: "EvaluationSettings"
#     logging_settings: "LoggingSettings"
#
#     # Class-level cache
#     _instance: Optional["MainSettings"] = None
#     _config_folder: Optional[Path] = None
#
#     def __init__(
#         self,
#         input_settings: "InputSettings",
#         module_settings: "ModuleSettings",
#         anonymization_settings: "AnonymizationSettings",
#         evaluation_settings: "EvaluationSettings",
#         logging_settings: "LoggingSettings",
#     ):
#         self.input_settings = input_settings
#         self.module_settings = module_settings
#         self.anonymization_settings = anonymization_settings
#         self.evaluation_settings = evaluation_settings
#         self.logging_settings = logging_settings
#
#     @classmethod
#     def configure(cls, config_folder: Path) -> None:
#         """Set the folder from which to load configs."""
#         cls._config_folder = config_folder
#
#     @classmethod
#     def get(cls) -> "MainSettings":
#         """Return the singleton instance, loading it if necessary."""
#         if cls._instance is None:
#             if cls._config_folder is None:
#                 raise RuntimeError("MainSettings not configured. Call configure() first.")
#
#             from your_settings_module import InputSettings, ModuleSettings, AnonymizationSettings, EvaluationSettings, LoggingSettings
#             from your_settings_module import create_settings_from_config_file
#
#             cls._instance = cls(
#                 input_settings=create_settings_from_config_file(cls._config_folder / "input_config.yaml", InputSettings),
#                 module_settings=create_settings_from_config_file(cls._config_folder / "modules_config.yaml", ModuleSettings),
#                 anonymization_settings=create_settings_from_config_file(cls._config_folder / "anonymization_config.yaml", AnonymizationSettings),
#                 evaluation_settings=create_settings_from_config_file(cls._config_folder / "evaluation_config.yaml", EvaluationSettings),
#                 logging_settings=create_settings_from_config_file(cls._config_folder / "logging_config.yaml", LoggingSettings),
#             )
#         return cls._instance
#
#
#
# def load_main_settings(config_folder: Path) -> MainSettings:
#     """
#     Load all main settings from a config folder containing multiple YAML files.
#     Args:
#         config_folder (Path): Path to folder with config files.
#     Returns:
#         MainSettings: Main settings object with all sub-settings loaded.
#     """
#     return MainSettings(
#         input_settings=create_settings_from_config_file(config_folder / "input_config.yaml", InputSettings),
#         module_settings=create_settings_from_config_file(config_folder / "modules_config.yaml", ModuleSettings),
#         anonymization_settings=create_settings_from_config_file(
#             config_folder / "anonymization_config.yaml", AnonymizationSettings
#         ),
#         evaluation_settings=create_settings_from_config_file(
#             config_folder / "evaluation_config.yaml", EvaluationSettings
#         ),
#         logging_settings=create_settings_from_config_file(config_folder / "logging_config.yaml", LoggingSettings),
#     )
#
#
# @dataclass
# class MainSettings:
#     input_folder: str
#     output_folder: str
#     face_detector_type: str
#     face_detector_model_path: str
#     face_detector_confidence: float = 0.3
#     facial_landmarks_detector_type: str = "spiga"
#     facial_landmarks_detector_model_path: str = "models/spiga.pth"
#     anonymization_method: str = "black_box"
#     log_level: str = "info"
#
#     # Internal defaults
#     save_face_detection_visualization: bool = field(default=False)
#     save_facial_landmarks_visualization: bool = field(default=False)
#     max_frame_detection_lookback: int = field(default=0)
#     anonymization_padding_method: str = field(default="ratio")
#     anonymization_padding_ratio: float = field(default=0.75)
#     anonymization_padding_constant: int = field(default=96)
#
#     @staticmethod
#     def from_configs(config_path: Path, defaults_path: Path) -> "MainSettings":
#         """
#         Load main settings from config and defaults YAML files.
#         Args:
#             config_path (Path): Path to main config YAML.
#             defaults_path (Path): Path to defaults YAML.
#         Returns:
#             MainSettings: Loaded settings object.
#         """
#         with open(config_path, "r") as f:
#             config = yaml.safe_load(f)
#         with open(defaults_path, "r") as f:
#             defaults = yaml.safe_load(f)
#         # Flatten nested detector configs
#         face_detector = config.get("face_detector", {})
#         facial_landmarks_detector = config.get("facial_landmarks_detector", {})
#         return MainSettings(
#             input_folder=config.get("input_folder"),
#             output_folder=config.get("output_folder"),
#             face_detector_type=face_detector.get("type"),
#             face_detector_model_path=face_detector.get("model_path"),
#             face_detector_confidence=face_detector.get("confidence", 0.3),
#             facial_landmarks_detector_type=facial_landmarks_detector.get("type", "spiga"),
#             facial_landmarks_detector_model_path=facial_landmarks_detector.get("model_path", "models/spiga.pth"),
#             anonymization_method=config.get("anonymization_method", "black_box"),
#             log_level=config.get("log_level", "info"),
#             save_face_detection_visualization=defaults.get("save_face_detection_visualization", False),
#             save_facial_landmarks_visualization=defaults.get("save_facial_landmarks_visualization", False),
#             max_frame_detection_lookback=defaults.get("max_frame_detection_lookback", 0),
#             anonymization_padding_method=defaults.get("anonymization_padding_method", "ratio"),
#             anonymization_padding_ratio=defaults.get("anonymization_padding_ratio", 0.75),
#             anonymization_padding_constant=defaults.get("anonymization_padding_constant", 96),
#         )
