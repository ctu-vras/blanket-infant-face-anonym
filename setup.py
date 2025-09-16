from setuptools import setup, find_packages

setup(
    name="blanket",
    version="0.1.0",
    description="BLANKET: Anonymizing Faces in Infant Video Recordings",
    author="Ditmar Hadera, Jan Cech, Miroslav Purkrabek, Matej Hoffmann",
    packages=find_packages(include=["blanket*"]),
    install_requires=[
        "torch>=1.12",
        "ultralytics",
        "numpy",
        "opencv-python",
        "pyyaml"
    ],
    python_requires=">=3.9",
    include_package_data=True,
    package_data={
        "blanket": [
            "configs/*.yaml",
            "configs/defaults.yaml",
            "configs/config.yaml",
            "configs/detector_parameters/face_detector_parameters/*.yaml",
            "configs/detector_parameters/facial_landmarks_detector_parameters/*.yaml",
            "configs/module_parameters/*.yaml",
            "configs/module_parameters/*.json",
            "constants/enums/*.py",
            "core/objects/*.py",
            "core/detectors/face_detectors/*.py",
            "core/detectors/facial_landmarks_detectors/*.py",
            "anonymization/methods/*.py",
            "anonymization/pipelines/*.py",
            "core/geometry.py",
            "core/visualization.py"
        ]
    },
    entry_points={
        "console_scripts": [
            "blanket-run-image=run_image_anonymization:main"
        ]
    },
    license="GPL-3.0"
)
