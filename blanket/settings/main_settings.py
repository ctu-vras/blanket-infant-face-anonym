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
