
import os
from pathlib import Path
import cv2
import numpy as np
from source.settings.main_settings import MainSettings
from source.constants.enums.detection_enums import FaceDetectorModule
from source.core.detectors.detector_factory import DetectorFactory

from source.anonymization.methods.black_box import BlackBoxAnonymizer
from source.anonymization.methods.pixelation import PixelationAnonymizer
from source.anonymization.methods.gaussian_blur import GaussianBlurAnonymizer

# Load consolidated config
CONFIG_DIR = Path(os.path.join(os.path.dirname(__file__), 'configs'))
main_settings = MainSettings.from_configs(CONFIG_DIR / 'config.yaml', CONFIG_DIR / 'defaults.yaml')

input_folder = main_settings.input_folder
output_folder = Path(main_settings.output_folder)
os.makedirs(output_folder, exist_ok=True)

# Instantiate detectors
face_detector = DetectorFactory.create_face_detector(FaceDetectorModule(main_settings.face_detector_type))

black_box = BlackBoxAnonymizer()
pixelation = PixelationAnonymizer()
gaussian_blur = GaussianBlurAnonymizer()

# Process images in input folder

for img_name in os.listdir(input_folder):
    img_path = Path(input_folder) / img_name
    if not img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
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
