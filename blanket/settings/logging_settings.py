from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoggingSettings:
    process_console_log_level: str = "INFO"
    process_file_log_level: str = "DEBUG"
    process_log_filepath: Path = Path("logs/process_log.txt")

    log_anonymization_result: bool = True
    anonymization_result_log_filepath: Path = Path("logs/anonymization_results.txt")

    log_frames_without_detections: bool = True
    frames_without_detections_log_filepath: Path = Path("logs/frames_without_detections.txt")
