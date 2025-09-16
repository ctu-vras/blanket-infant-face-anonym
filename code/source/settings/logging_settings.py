from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import logging
import logging.config
import os
import yaml
# from logging_constants import (PROCESS_LOGGER_NAME,
#                                ANONYMIZATION_RESULT_LOGGER_NAME,
#                                FRAMES_WITHOUT_DETECTIONS_LOGGER_NAME)

PROCESS_LOGGER_NAME = "process_logger"
ANONYMIZATION_RESULT_LOGGER_NAME = "anonymization_result_logger"
FRAMES_WITHOUT_DETECTIONS_LOGGER_NAME = "frames_without_detections_logger"



@dataclass
class LoggingSettings:
    process_console_log_level: str = "INFO"
    process_file_log_level: str = "DEBUG"
    process_log_filepath: Path = Path("logs/process_log.txt")

    log_anonymization_result: bool = True
    anonymization_result_log_filepath: Path = Path("logs/anonymization_results.txt")

    log_frames_without_detections: bool = True
    frames_without_detections_log_filepath: Path = Path("logs/frames_without_detections.txt")

    process_logger: logging.Logger = field(init=False)
    anonymization_result_logger: logging.Logger = field(init=False)
    frames_without_detections_logger: logging.Logger = field(init=False)

    def __post_init__(self):
        self._setup_process_logger()
        if self.log_anonymization_result:
            self._setup_result_logger()
        if self.log_frames_without_detections:
            self._setup_frames_without_detections_logger()

    def _setup_process_logger(self):
        self.process_logger = logging.getLogger(PROCESS_LOGGER_NAME)
        self.process_logger.setLevel(logging.DEBUG)  # catch everything

        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        file_formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

        # console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.process_console_log_level.upper())
        console_handler.setFormatter(console_formatter)
        self.process_logger.addHandler(console_handler)

        # file handler
        os.makedirs(self.process_log_filepath.parent, exist_ok=True)
        file_handler = logging.FileHandler(self.process_log_filepath, mode="w")
        file_handler.setLevel(self.process_file_log_level.upper())
        file_handler.setFormatter(file_formatter)
        self.process_logger.addHandler(file_handler)

        self.process_logger.propagate = False  # stops messages from being propagated to ancestor loggers

    def _setup_result_logger(self):
        self.anonymization_result_logger = logging.getLogger(ANONYMIZATION_RESULT_LOGGER_NAME)
        self.anonymization_result_logger.setLevel(logging.INFO)

        os.makedirs(self.anonymization_result_log_filepath.parent, exist_ok=True)
        file_handler = logging.FileHandler(self.anonymization_result_log_filepath, mode="w")
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        self.anonymization_result_logger.addHandler(file_handler)
        self.anonymization_result_logger.propagate = False

    def _setup_frames_without_detections_logger(self):
        self.frames_without_detections_logger = logging.getLogger(FRAMES_WITHOUT_DETECTIONS_LOGGER_NAME)
        self.frames_without_detections_logger.setLevel(logging.INFO)

        os.makedirs(self.frames_without_detections_log_filepath.parent, exist_ok=True)
        file_handler = logging.FileHandler(self.frames_without_detections_log_filepath, mode="w")
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        self.frames_without_detections_logger.addHandler(file_handler)
        self.frames_without_detections_logger.propagate = False

    def log_frame_without_detection(self, frame_number: int):
        if self.log_frames_without_detections:
            self.frames_without_detections_logger.info(f"Frame {frame_number:06d} - no faces detected.")
