import os
from pathlib import Path

import cv2
import numpy as np

from blanket.constants.paths import CONFIG_FOLDER
from blanket.settings.main_settings import MainSettings
from blanket.core.logging_setup import setup_loggers
from blanket.anonymization.anonymizers.black_box import BlackBoxAnonymizer
from blanket.anonymization.anonymizers.gaussian_blur import GaussianBlurAnonymizer
from blanket.anonymization.anonymizers.pixelation import PixelationAnonymizer
from blanket.core.objects.primitives import ImagePrimitive
from blanket.constants.enums.detection_enums import FaceDetectorModule
from blanket.core.detectors.detector_factory import DetectorCache
from blanket.constants.enums.anonymization_enums import AnonymizationMethod
from blanket.anonymization.anonymizer_factory import AnonymizerCache


# basic setup
MainSettings.configure(CONFIG_FOLDER / "config.yaml")

main_settings = MainSettings.get()

setup_loggers(main_settings.logging_settings)

input_folder = main_settings.input_settings.input_folder
output_folder = Path(main_settings.input_settings.output_folder)
os.makedirs(output_folder, exist_ok=True)

# Instantiate detectors
face_detector = DetectorCache.get_face_detector(FaceDetectorModule(main_settings.module_settings.face_detector_name))
anonymizer = AnonymizerCache.get_anonymizer(AnonymizationMethod(main_settings.module_settings.anonymizer_name))

# Process images in input folder
for img_name in os.listdir(input_folder):
    img_path = Path(input_folder) / img_name
    if not img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        continue
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"Failed to load {img_path}")
        continue
    # Detect faces
    detections = face_detector.detect(image)

    # Visualize detections
    vis_image = image.copy()
    for det in detections:
        x, y, w, h = det.left_top_width_height
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Run anonymizer
    original_image = ImagePrimitive(image)
    anonymized_image = anonymizer.anonymize_image(original_image, detections).image_bgr

    # Stitch in chessboard pattern (2x2)
    image_comparison = np.vstack([vis_image, anonymized_image])

    # Save
    out_path = os.path.join(output_folder, img_name)
    cv2.imwrite(str(out_path), image_comparison)
    print(f"Saved anonymized image to {out_path}")
