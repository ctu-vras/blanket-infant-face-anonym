# Adding a New Detector

This guide explains how to add a new **face detector** or **facial landmarks detector** to the project.

---

## Steps

### 1. Add a corresponding enum entry
Locate the enums that define supported detectors (e.g. `FaceDetectorModule`, `FacialLandmarksDetectorModule`) in 
`code/source/constants/enums/detection_enums.py` and add your new detector:

```python
class FaceDetectorModule(Enum):
    YOLO = "yolo"
    MY_NEW_FACE_DETECTOR = "my_new_face_detector"  # <- add here
```

### 2. Implement the detector class
Extend the detectors folder `code/source/core/detectors/...` with your detector implementation.
Your class should inherit from the correct base class:

#### Face Detector Example
```python
import numpy as np
from blanket.core.objects.detections import FaceDetection
from blanket.settings.individual_modules_settings.face_detector_settings import FaceDetectorSettings
from blanket.core.detectors.base_detectors import BaseFaceDetector

class MyNewFaceDetector(BaseFaceDetector):
    def __init__(self, settings: FaceDetectorSettings):
        super().__init__(settings)
        # ...
    
    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]:
        # implement face detection logic
        detections: list[FaceDetection] = []
        # ...
        return detections
```

#### Facial Landmarks Detector Example
```python
import numpy as np
from blanket.core.objects.detections import FaceDetection, FacialLandmarksDetection
from blanket.settings.individual_modules_settings.facial_landmarks_detector_settings import FacialLandmarksDetectorSettings 
from blanket.core.detectors.base_detectors import BaseFacialLandmarksDetector

class MyNewLandmarksDetector(BaseFacialLandmarksDetector):
    def __init__(self, settings: FacialLandmarksDetectorSettings):
        super().__init__(settings)
        # ...
    
    def detect(self, image_bgr: np.ndarray, face_detection: FaceDetection) -> FacialLandmarksDetection:
        # implement landmarks detection logic
        landmarks: FacialLandmarksDetection = ...
        return landmarks
```

### 3. Add a parameter YAML file
Each detector has its own configuration YAML file under `code/configs/detector_parameters/...`.
This file stores detector-specific settings such as thresholds, model paths, or backend preferences.

```yaml
confidence_threshold: 0.5
model_path: "models/my_new_detector.onnx"
use_gpu: true
```

#### 4. Register the detector in the factory
Update `DetectorFactory` in `code/source/core/detectors/detector_factory.py` so it knows how to create your new detector:

```python
face_detector_parameters_folder = Path("configs/detector_parameters/face_detector_parameters")

face_detector_registry: dict[FaceDetectorModule, tuple[Type[BaseFaceDetector], Path]] = {
    FaceDetectorModule.YOLO: (
        YOLOFaceDetector, Path("yolo_parameters.yaml")),
    FaceDetectorModule.MY_NEW_FACE_DETECTOR: (
        MyNewFaceDetector, Path("my_new_face_detector.yaml")),  # <- add here
}


facial_landmarks_detector_parameters_folder = Path("configs/detector_parameters/facial_landmarks_detector_parameters")

facial_landmarks_detector_registry: dict[FacialLandmarksDetectorModule, tuple[Type[BaseFacialLandmarksDetector], Path]] = {
    FacialLandmarksDetectorModule.SPIGA: (
        SPIGAFacialLandmarksDetector, Path("spiga_parameters.yaml")),
    FacialLandmarksDetectorModule.MY_NEW_FACIAL_LANDMARKS_DETECTOR: (
        MyNewLandmarksDetector, Path("my_new_facial_landmarks_detector.yaml")),  # <- add here
}
```
