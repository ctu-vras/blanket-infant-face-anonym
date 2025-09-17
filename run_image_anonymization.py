import os
from pathlib import Path

import cv2
import numpy as np

from blanket.constants.paths import CONFIG_FOLDER
from blanket.settings.main_settings import MainSettings
from blanket.core.logging_setup import setup_loggers
# from blanket.anonymization.anonymizers.black_box import BlackBoxAnonymizer
# from blanket.anonymization.anonymizers.gaussian_blur import GaussianBlurAnonymizer
# from blanket.anonymization.anonymizers.pixelation import PixelationAnonymizer
# from blanket.constants.enums.detection_enums import FaceDetectorModule
# from blanket.core.detectors.detector_factory import DetectorFactory


# basic setup
MainSettings.configure(CONFIG_FOLDER / "config.yaml")
setup_loggers(MainSettings.get().logging_settings)




input_folder = MainSettings.get().input_settings.input_folder
output_folder = Path(MainSettings.get().input_settings.output_folder)
os.makedirs(output_folder, exist_ok=True)

# Instantiate detectors
# face_detector = DetectorFactory.create_face_detector(FaceDetectorModule(main_settings.face_detector_type))
#
# black_box = BlackBoxAnonymizer()
# pixelation = PixelationAnonymizer()
# gaussian_blur = GaussianBlurAnonymizer()

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

    # Run all anonymizations
    img_black_box = black_box.anonymize(image, detections)
    img_pixelation = pixelation.anonymize(image, detections)
    img_gaussian = gaussian_blur.anonymize(image, detections)

    # Stitch in chessboard pattern (2x2)
    top_row = np.hstack([vis_image, img_black_box])
    bottom_row = np.hstack([img_gaussian, img_pixelation])
    chessboard = np.vstack([top_row, bottom_row])

    # Save
    out_path = os.path.join(output_folder, img_name)
    cv2.imwrite(str(out_path), chessboard)
    print(f"Saved anonymized image to {out_path}")
