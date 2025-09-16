from dataclasses import fields
from pathlib import Path
from typing import Type, TypeVar

import yaml

from .anonymization_settings import AnonymizationSettings
from .evaluation_settings import EvaluationSettings
from .input_settings import InputSettings
from .logging_settings import LoggingSettings
from .main_settings import MainSettings
from .module_settings import ModuleSettings

# T = TypeVar("T", bound=BaseSettings)  # with all settings classes inheriting from class BaseSettings(ABC)
T = TypeVar("T")


def create_settings_from_config_file(config_filepath: Path, settings_class: Type[T]) -> T:
    """
    Load settings from a YAML config file, requiring all keys to match dataclass fields.
    Args:
        config_filepath (Path): Path to YAML config file.
        settings_class (Type[T]): Dataclass type to instantiate.
    Returns:
        T: Instance of settings_class with loaded values.
    """
    with open(config_filepath, "r") as config_file:
        config_dict = yaml.safe_load(config_file)
    return settings_class(**config_dict)


def create_settings_with_extras_from_config_file(config_filepath: Path, settings_class: Type[T]) -> T:
    """
    Load settings from a YAML config file, allowing unknown keys to be stored in 'extra_parameters'.
    Args:
        config_filepath (Path): Path to YAML config file.
        settings_class (Type[T]): Dataclass type to instantiate.
    Returns:
        T: Instance of settings_class with loaded values and extras.
    """
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


def load_main_settings(config_folder: Path) -> MainSettings:
    """
    Load all main settings from a config folder containing multiple YAML files.
    Args:
        config_folder (Path): Path to folder with config files.
    Returns:
        MainSettings: Main settings object with all sub-settings loaded.
    """
    return MainSettings(
        input_settings=create_settings_from_config_file(config_folder / "input_config.yaml", InputSettings),
        module_settings=create_settings_from_config_file(config_folder / "modules_config.yaml", ModuleSettings),
        anonymization_settings=create_settings_from_config_file(
            config_folder / "anonymization_config.yaml", AnonymizationSettings
        ),
        evaluation_settings=create_settings_from_config_file(
            config_folder / "evaluation_config.yaml", EvaluationSettings
        ),
        logging_settings=create_settings_from_config_file(config_folder / "logging_config.yaml", LoggingSettings),
    )
