from typing import List, Optional
from pathlib import Path
import subprocess

from blanket.core.objects.primitives import ImagePrimitive, VideoPrimitive
from blanket.core.objects.detections import FaceDetection, FacialLandmarksDetection
from blanket.anonymization.base_anonymizer import BaseAnonymizer


class FacefusionAnonymizer(BaseAnonymizer):
    def anonymize_image(
            self,
            input_image: ImagePrimitive,
            face_detections: Optional[List[FaceDetection]] = None,
            facial_landmarks_detection: Optional[List[FacialLandmarksDetection]] = None,
            conditioning: Optional[AnonymizationConditioning] = None
    ) -> ImagePrimitive:
    # def anonymize_image(self, image, detections):
        """
        Placeholder for Facefusion anonymization method.
        Args:
            image (np.ndarray): BGR image.
            detections (list): List of FaceDetection objects.
        Raises:
            NotImplementedError: Always, as not implemented yet.
        """
        raise NotImplementedError("Facefusion image anonymization not implemented yet.")

    def anonymize_video(self, input_video: VideoPrimitive, input_image: ImagePrimitive, output_path: Path) -> VideoPrimitive:
        # define command arguments for the running of facefusion using the subprocess module
        # TODO - move these into parameters
        facefusion_script = "facefusion.py"
        facefusion_arguments = [
            "headless-run",
            "-s", input_image.path,  # source path
            "-t", input_video.path,  # target path
            "-o", output_path,  # output path
            "--face-detector-model", "yoloface",
            "--face-detector-score", "0.25",
            "--face-landmarker-score", "0.15",
            "--face-selector-mode", "reference",
            "--face-mask-types", "box", "occlusion", "region",
            "--face-mask-blur", "0.60",
            "--output-image-quality", "100",
            "--output-video-preset", "medium",
            "--skip-audio",  # removing audio
            "--processors", "face_swapper",
            # "face_enhancer",  # "frame_enhancer", "lip_syncer", "face_debugger",
            # "--expression-restorer-factor", "80",
            # "--face-debugger-items", "bounding-box", "face-landmark-68", "face-detector-score", "face-landmarker-score",
            # "--face-enhancer-blend", "40",
            "--face-swapper-pixel-boost", "512x512",  # "768x768", "1024x1024",
            "--execution-providers", "cuda",
            # "--execution-device-id", "3",
            "--execution-thread-count", "1",
            # "--execution-queue-count", "1",
            "--log-level", "debug",
        ]

        # launch facefusion
        subprocess.run(["bash", "run_facefusion.sh", facefusion_script, *facefusion_arguments],
                       cwd="../../facefusion")  # blocking mode
        # process = subprocess.Popen(["bash", "run_facefusion.sh", facefusion_script, *facefusion_arguments],
        #                            cwd="../../facefusion")  # non-blocking mode

        return VideoPrimitive(output_path)
