import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import yaml
import torch
import gc

from blanket.anonymization.methods.stable_diffusion import StableDiffusionAnonymizer
from blanket.core.detectors.detector_factory import DetectorFactory
from blanket.constants.enums.detection_enums import FaceDetectorModule, FacialLandmarksDetectorModule


def generate_synthetic_identity(image, output_dir, device=None, identity_config_path=None, save_debug=False):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if identity_config_path is None:
        identity_config_path = Path(__file__).parent.parent.parent / "configs" / "module_parameters" / "stable_diffusion_parameters.yaml"

    with open(identity_config_path, 'r') as f:
        config = yaml.safe_load(f)

    use_poisson = config.get('use_poisson_blending', False)
    poisson_mode = config.get('poisson_blend_mode', 'NORMAL')

    face_detector = DetectorFactory.create_face_detector(FaceDetectorModule.YOLO)
    face_detections = face_detector.detect(image)

    if len(face_detections) == 0:
        raise RuntimeError("No face detected in the input image")

    face_detection = face_detections[0]
    face_bbox = face_detection.left_top_right_bottom

    landmarks_detector = DetectorFactory.create_facial_landmarks_detector(FacialLandmarksDetectorModule.SPIGA)
    landmarks_detection = landmarks_detector.detect(image, face_detection)
    face_landmarks = landmarks_detection.landmarks

    orig_h, orig_w = image.shape[:2]

    # free memory
    del face_detector
    del landmarks_detector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    anonymizer = StableDiffusionAnonymizer(config_path=identity_config_path, device=device)

    mask_path = None
    if save_debug:
        mask_path = str(output_path / "inpainting_mask.png")

    synthetic_image = anonymizer.generate(
        image=image,
        face_bbox=face_bbox,
        face_landmarks=face_landmarks,
        output_size=(orig_w, orig_h),
        save_mask_path=mask_path
    )

    if use_poisson and mask_path is not None:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        synthetic_np = np.array(synthetic_image)
        synthetic_bgr = cv2.cvtColor(synthetic_np, cv2.COLOR_RGB2BGR)

        moments = cv2.moments(mask)
        if moments['m00'] != 0:
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])
            center = (center_x, center_y)

            blend_flag = cv2.NORMAL_CLONE if poisson_mode == 'NORMAL' else cv2.MIXED_CLONE
            blended = cv2.seamlessClone(synthetic_bgr, image, mask, center, blend_flag)
            blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
            synthetic_image = Image.fromarray(blended_rgb)

    identity_path = str(output_path / "synthetic_identity.jpg")
    synthetic_image.save(identity_path)

    anonymizer.unload()

    return identity_path, mask_path


class ImagePipeline:
    def __init__(self, output_dir="output", device=None, identity_config_path=None, debug=False):
        self.output_dir = output_dir
        self.device = device
        self.identity_config_path = identity_config_path
        self.debug = debug

    def run(self, image_path):
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))

        identity_path, mask_path = generate_synthetic_identity(
            image=image,
            output_dir=self.output_dir,
            device=self.device,
            identity_config_path=self.identity_config_path,
            save_debug=self.debug,
        )

        return {
            "success": True,
            "identity_image": identity_path,
            "mask_image": mask_path,
        }

