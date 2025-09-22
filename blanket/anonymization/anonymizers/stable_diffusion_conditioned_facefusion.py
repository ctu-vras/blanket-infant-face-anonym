from typing import List, Optional
from pathlib import Path
import os
import tempfile

from blanket.core.objects.primitives import ImagePrimitive, VideoPrimitive
from blanket.core.objects.detections import FaceDetection, FacialLandmarksDetection
from blanket.settings.main_settings import MainSettings
from blanket.constants.enums.detection_enums import FaceDetectorModule
from blanket.core.detectors.base_detectors import BaseFaceDetector
from blanket.core.detectors.detector_factory import DetectorCache
from blanket.constants.enums.anonymization_enums import AnonymizationMethod
from blanket.anonymization.base_anonymizer import BaseAnonymizer
from blanket.anonymization.anonymizers.stable_diffusion import StableDiffusionAnonymizer
from blanket.anonymization.anonymizers.facefusion import FacefusionAnonymizer
# from blanket.anonymization.anonymizer_factory import AnonymizerCache


class StableDiffusionConditionedFacefusionAnonymizer(BaseAnonymizer):
    def anonymize_image(
            self,
            input_image: ImagePrimitive,
            face_detections: Optional[List[FaceDetection]] = None,
            facial_landmarks_detection: Optional[List[FacialLandmarksDetection]] = None,
            # conditioning: Optional[AnonymizationConditioning] = None
    ) -> ImagePrimitive:
        """
        Placeholder for Stable Diffusion Conditioned Facefusion anonymization method.
        Args:
            image (np.ndarray): BGR image.
            detections (list): List of FaceDetection objects.
        Raises:
            NotImplementedError: Always, as not implemented yet.
        """
        raise NotImplementedError("Stable diffusion conditioned FaceFusion image anonymization not implemented yet.")

    def anonymize_video(self, input_video: VideoPrimitive, output_path: Path) -> VideoPrimitive:
        raise NotImplementedError("Stable diffusion conditioned FaceFusion video anonymization not implemented yet.")
        # filename, _ = os.path.splitext(os.path.basename(input_video.path))
        #
        # settings = MainSettings.get()
        # # get auxiliary anonymizers
        # stable_diffusion_anonymizer: StableDiffusionAnonymizer = AnonymizerCache.get_anonymizer(AnonymizationMethod.STABLE_DIFFUSION)
        # facefusion_anonymizer: FacefusionAnonymizer = AnonymizerCache.get_anonymizer(AnonymizationMethod.FACEFUSION)
        #
        # with input_video as original_video:
        #     # try to get some decent frame anonymized using StableDiffusion to work as source for FaceFusion
        #     for frame_index, original_frame in enumerate(original_video):
        #         anonymized_frame = stable_diffusion_anonymizer.anonymize_image(original_frame)
        #
        #         if self._is_frame_decently_anonymized(anonymized_frame):
        #             # temporarily save the anonymized frame
        #             # TODO - try using tempfile
        #             anonymized_frame.path = ...
        #             anonymized_frame.save_image(
        #                 rotate_back=False,
        #                 restricted_access_to_saved=settings.input_settings.restrict_access_to_saved
        #             )
        #
        #             facefusion_anonymizer.anonymize_video(input_video, anonymized_frame, output_path)
        #
        #             # TODO - check that the FaceFusion anonymization finished successfully
        #             if self._check_facefusion_success(output_path):
        #                 break
        #
        #     else:  # failed to get any decent frame anonymized by StableDiffusion
        #         pass

    def _is_frame_decently_anonymized(self, anonymized_frame: ImagePrimitive) -> bool:
        settings = MainSettings.get()
        face_detector: BaseFaceDetector = DetectorCache.get_face_detector(
            FaceDetectorModule(settings.module_settings.face_detector_name))

        # anonymized face should also be detectable
        # self.settings.extra_parameters["minimum_redetection_confidence"]

        return True

    def _check_facefusion_success(self, output_path: Path) -> bool:
        return True
