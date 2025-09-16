from dataclasses import dataclass

from .input_settings import InputSettings
from .module_settings import ModuleSettings
from .anonymization_settings import AnonymizationSettings
from .evaluation_settings import EvaluationSettings
from .logging_settings import LoggingSettings


@dataclass
class MainSettings:
    input_settings: InputSettings
    module_settings: ModuleSettings
    anonymization_settings: AnonymizationSettings
    evaluation_settings: EvaluationSettings
    logging_settings: LoggingSettings
