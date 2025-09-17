from __future__ import annotations
from typing import Optional, List
from pathlib import Path
from abc import ABC, abstractmethod
import cv2

from blanket.settings.module_settings.anonymizer_settings import AnonymizerSettings
from blanket.core.objects.primitives import ImagePrimitive, VideoPrimitive
from blanket.core.objects.detections import FaceDetection, FacialLandmarksDetection


class BaseAnonymizer(ABC):
    requires_face_detections: bool
    requires_facial_landmarks_detections: bool

    def __init__(self, settings: AnonymizerSettings):
        self.settings = settings

    @abstractmethod
    def anonymize_image(
            self,
            input_image: ImagePrimitive,
            face_detections: Optional[List[FaceDetection]] = None,
            facial_landmarks_detection: Optional[List[FacialLandmarksDetection]] = None,
            conditioning: Optional[AnonymizationConditioning] = None
    ) -> ImagePrimitive:
        """
        Anonymize a single image (frame).
        Frame-by-frame anonymizers must implement this.
        """
        pass

    def anonymize_video(self, input_video: VideoPrimitive, output_path: Path) -> VideoPrimitive:
        """
        Default implementation: read video frame-by-frame,
        apply anonymize_image(), and write to output.
        Methods that don't support frame-by-frame should override this.
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_video_writer = cv2.VideoWriter(
            str(output_path), fourcc, input_video.fps, (input_video.width, input_video.height))

        with input_video as original_video:
            for frame_index, original_frame in enumerate(original_video):
                # TODO - create additional inputs for anonymize_image() if required

                anonymized_frame = self.anonymize_image(original_frame)

                # rotate back to original orientation
                anonymized_frame = anonymized_frame.to_not_rotated_image()

                output_video_writer.write(anonymized_frame.image_bgr)

        output_video_writer.release()

        return VideoPrimitive(output_path, clockwise_rotation_index=0)
