import os
from pathlib import Path
import cv2
from source.settings.config_loader import create_settings_from_config_file
from source.settings.input_settings import InputSettings
from source.settings.module_settings import ModuleSettings
from source.core.detectors.detector_factory import DetectorFactory
from source.anonymization.methods.black_box import BlackBoxAnonymizer

# Load config
CONFIG_DIR = Path(os.path.join(os.path.dirname(__file__), 'configs'))
input_settings = create_settings_from_config_file(CONFIG_DIR / 'input_config.yaml', InputSettings)
module_settings = create_settings_from_config_file(CONFIG_DIR / 'modules_config.yaml', ModuleSettings)

input_folder = input_settings.input_folder
output_folder = Path('outputs')
os.makedirs(output_folder, exist_ok=True)

# Instantiate detectors
# breakpoint()
face_detector = DetectorFactory.create_face_detector(module_settings.face_detector_name)
# landmarks_detector = DetectorFactory.create_facial_landmarks_detector(module_settings.facial_landmarks_detector_name)
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
    # Detect landmarks for each face
    # for det in detections:
    #     det.landmarks = landmarks_detector.detect(image, det)
    # Anonymize
    anonymized = anonymizer.anonymize(image, detections)
    # Save
    out_path = output_folder / img_name
    cv2.imwrite(str(out_path), anonymized)
    print(f"Saved anonymized image to {out_path}")
