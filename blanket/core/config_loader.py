from dataclasses import fields
from pathlib import Path
from typing import Type, TypeVar

import yaml


# T = TypeVar("T", bound=BaseSettings)  # with all settings classes inheriting from class BaseSettings(ABC)
T = TypeVar("T")


def config_file_to_dictionary(config_filepath: Path) -> dict:
    with open(config_filepath, "r") as config_file:
        config_dict = yaml.safe_load(config_file)

    if config_dict is not None:
        return config_dict
    else:
        return {}


def create_settings_from_config_file(config_filepath: Path, settings_class: Type[T]) -> T:
    """
    Load settings from a YAML config file, requiring all keys to match dataclass fields.
    Args:
        config_filepath (Path): Path to YAML config file.
        settings_class (Type[T]): Dataclass type to instantiate.
    Returns:
        T: Instance of settings_class with loaded values.
    """
    return settings_class(**config_file_to_dictionary(config_filepath))


def create_settings_with_extras_from_config_file(config_filepath: Path, settings_class: Type[T]) -> T:
    """
    Load settings from a YAML config file, allowing unknown keys to be stored in 'extra_parameters'.
    Args:
        config_filepath (Path): Path to YAML config file.
        settings_class (Type[T]): Dataclass type to instantiate.
    Returns:
        T: Instance of settings_class with loaded values and extras.
    """
    # TODO - check that config file exists
    config_dict = config_file_to_dictionary(config_filepath)

    field_names = {field.name for field in fields(settings_class)}
    known_parameters, extra_parameters = {}, {}

    for key, value in config_dict.items():
        if key in field_names:
            known_parameters[key] = value
        else:
            extra_parameters[key] = value

    return settings_class(**known_parameters, extra_parameters=extra_parameters)
