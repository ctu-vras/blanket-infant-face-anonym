from pathlib import Path
import os
import logging

from blanket.constants.logging_constants import (PROCESS_LOGGER_NAME, ANONYMIZATION_RESULT_LOGGER_NAME,
                                                 FRAMES_WITHOUT_DETECTIONS_LOGGER_NAME)
from blanket.settings.logging_settings import LoggingSettings


def setup_loggers(settings: LoggingSettings) -> dict[str, logging.Logger]:
    loggers = {}

    # process logger
    process_logger = logging.getLogger(PROCESS_LOGGER_NAME)

    if not process_logger.handlers:
        process_logger.setLevel(logging.DEBUG)

        _add_console_handler_to_logger(process_logger, settings.process_console_log_level)
        # TODO - check that the path is interpreted correctly
        _add_file_handler_to_logger(process_logger, Path(settings.process_log_filepath), settings.process_file_log_level)

        process_logger.propagate = False
        loggers[PROCESS_LOGGER_NAME] = process_logger

    # anonymization result logger
    if settings.log_anonymization_result:
        anonymization_result_logger = logging.getLogger(ANONYMIZATION_RESULT_LOGGER_NAME)

        if not anonymization_result_logger.handlers:
            anonymization_result_logger.setLevel(logging.DEBUG)
            _add_file_handler_to_logger(anonymization_result_logger,
                                        Path(settings.anonymization_result_log_filepath), "INFO", "%(message)s")

            anonymization_result_logger.propagate = False

        loggers[ANONYMIZATION_RESULT_LOGGER_NAME] = anonymization_result_logger

    # frames-without-detections logger
    if settings.log_frames_without_detections:
        frames_without_detections_logger = logging.getLogger(FRAMES_WITHOUT_DETECTIONS_LOGGER_NAME)

        if not frames_without_detections_logger.handlers:
            frames_without_detections_logger.setLevel(logging.DEBUG)
            _add_file_handler_to_logger(frames_without_detections_logger,
                                        Path(settings.frames_without_detections_log_filepath), "INFO", "%(message)s")
            frames_without_detections_logger.propagate = False

        loggers[FRAMES_WITHOUT_DETECTIONS_LOGGER_NAME] = frames_without_detections_logger

    return loggers


def _add_file_handler_to_logger(logger: logging.Logger, filepath: Path, level: str = "DEBUG",
                                fmt: str = "[%(asctime)s] %(levelname)s: %(message)s") -> None:
    os.makedirs(filepath.parent, exist_ok=True)

    handler = logging.FileHandler(filepath, mode="w")
    handler.setLevel(level.upper())
    handler.setFormatter(logging.Formatter(fmt))

    logger.addHandler(handler)


def _add_console_handler_to_logger(logger: logging.Logger, level: str = "INFO",
                                   fmt: str = "%(levelname)s: %(message)s") -> None:
    handler = logging.StreamHandler()
    handler.setLevel(level.upper())
    handler.setFormatter(logging.Formatter(fmt))

    logger.addHandler(handler)


# def log_frame_without_detection(self, frame_number: int):
#     """
#     Log a frame number where no faces were detected.
#     Args:
#         frame_number (int): Frame index.
#     """
#     if self.log_frames_without_detections:
#         self.frames_without_detections_logger.info(f"Frame {frame_number:06d} - no faces detected.")
