
import os
from pathlib import Path
import cv2
from source.settings.main_settings import MainSettings
from source.constants.enums.detection_enums import FaceDetectorModule
from source.core.detectors.detector_factory import DetectorFactory
from source.anonymization.methods.black_box import BlackBoxAnonymizer

# Load consolidated config
CONFIG_DIR = Path(os.path.join(os.path.dirname(__file__), 'configs'))
main_settings = MainSettings.from_configs(CONFIG_DIR / 'config.yaml', CONFIG_DIR / 'defaults.yaml')

input_folder = main_settings.input_folder
output_folder = Path(main_settings.output_folder)
os.makedirs(output_folder, exist_ok=True)

# Instantiate detectors
face_detector = DetectorFactory.create_face_detector(FaceDetectorModule(main_settings.face_detector_type))
anonymizer = BlackBoxAnonymizer()

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
    # Anonymize
    anonymized = anonymizer.anonymize(image, detections)
    # Save
    out_path = output_folder / img_name
    cv2.imwrite(str(out_path), anonymized)
    print(f"Saved anonymized image to {out_path}")
