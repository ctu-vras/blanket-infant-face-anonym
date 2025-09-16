import yaml
from pathlib import Path
from typing import Type, TypeVar
from dataclasses import fields

from main_settings import MainSettings
from input_settings import InputSettings
from module_settings import ModuleSettings
from anonymization_settings import AnonymizationSettings
from evaluation_settings import EvaluationSettings
from logging_settings import LoggingSettings


# T = TypeVar("T", bound=BaseSettings)  # with all settings classes inheriting from class BaseSettings(ABC)
T = TypeVar("T")


def create_settings_from_config_file(config_filepath: Path, settings_class: Type[T]) -> T:
    """Strict YAML -> SettingsClass loader. All YAML keys must match dataclass fields."""
    with open(config_filepath, "r") as config_file:
        config_dict = yaml.safe_load(config_file)
    return settings_class(**config_dict)


def create_settings_with_extras_from_config_file(config_filepath: Path, settings_class: Type[T]) -> T:
    """Flexible YAML -> SettingsClass loader. Unknown keys go into 'extra_parameters' field."""

    with open(config_filepath, "r") as config_file:
        config_dict = yaml.safe_load(config_file)

    field_names = {field.name for field in fields(settings_class)}
    known_parameters, extra_parameters = {}, {}

    for key, value in config_dict.items():
        if key in field_names:
            known_parameters[key] = value
        else:
            extra_parameters[key] = value

    return settings_class(**known_parameters, extra_parameters=extra_parameters)


# TODO - move this elsewhere
def load_main_settings(config_folder: Path) -> MainSettings:
    return MainSettings(
        input_settings=create_settings_from_config_file(config_folder / "input_config.yaml", InputSettings),
        module_settings=create_settings_from_config_file(config_folder / "module_config.yaml", ModuleSettings),
        anonymization_settings=create_settings_from_config_file(config_folder / "anonymization_config.yaml", AnonymizationSettings),
        evaluation_settings=create_settings_from_config_file(config_folder / "evaluation_config.yaml", EvaluationSettings),
        logging_settings=create_settings_from_config_file(config_folder / "logging_settings.yaml", LoggingSettings)
    )
