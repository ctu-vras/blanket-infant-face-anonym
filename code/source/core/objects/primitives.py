# TODO - define the classes that will be used to encapsule information on anonymization primitives (images and videos)
#  so that they can be used for both anonymization itself and then evaluation of the anonymization quality
#  supported features:
#      * rotating the primitive for the detection/anonymization/evaluation process and then rotating back before saving
#      * saving different stages of the detection/anonymization/evaluation process
#      * batch work on multiple primitives


from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
# from abc import ABC, abstractmethod
import os
import numpy as np
import cv2


# @dataclass
# class Primitive(ABC):
#     """Base class for a single media primitive (image or video)."""
#     path: Optional[Path] = None
#     rotation_index: int = 0
#     metadata: Dict[str, Any] = field(default_factory=dict)
#
#     @abstractmethod
#     def load(self) -> Any:
#         """Load the primitive from disk (numpy array, video reader, etc.)."""
#         ...
#
#     @abstractmethod
#     def save(self, path: Path, rotate_back: bool = True) -> None:
#         """Save the primitive back to disk."""
#         ...
#
#     @staticmethod
#     def rotate_image(image: np.ndarray, clockwise_index: int) -> Optional[np.ndarray]:
#         """Rotate an image array by 90° increments clockwise. For counterclockwise rotation use negative indices."""
#         if image is None:
#             raise ValueError("Cannot rotate None image")
#
#         clipped_clockwise_index = clockwise_index % 4
#
#         if clipped_clockwise_index == 0:
#             return image
#         elif clipped_clockwise_index == 1:
#             return cv2.rotate(cv2.ROTATE_90_CLOCKWISE, )
#         elif clipped_clockwise_index == 2:
#             return cv2.rotate(cv2.ROTATE_180, )
#         elif clipped_clockwise_index == 3:
#             return cv2.rotate(cv2.ROTATE_90_COUNTERCLOCKWISE, )
#         raise ValueError(f"Invalid rotation index {clockwise_index}")


@dataclass
class ImagePrimitive:
    """Represents a single image."""
    image_bgr: np.ndarray  # already rotated image in BGR format
    clockwise_rotation_index: int = 0  # index tracking in how many 90° increments has the image been rotated
    path: Optional[Path] = None

    @staticmethod
    def from_not_rotated_image(image_bgr: np.ndarray, clockwise_rotation_index: int = 0, path: Optional[Path] = None
                               ) -> ImagePrimitive:
        if clockwise_rotation_index:
            image_bgr = ImagePrimitive.rotate_image(image_bgr, clockwise_rotation_index)

        return ImagePrimitive(image_bgr, clockwise_rotation_index, path)

    @staticmethod
    def from_path(path: Path, clockwise_rotation_index: int = 0) -> ImagePrimitive:
        image_bgr = cv2.imread(str(path))

        return ImagePrimitive.from_not_rotated_image(image_bgr, clockwise_rotation_index, path)

    @property
    def image_rgb(self) -> np.ndarray:
        """Return the image in RGB format."""
        return self.convert_bgr_to_rgb(self.image_bgr)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.image_bgr.shape

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def height(self) -> int:
        return self.shape[0]

    def save_image(self, path: Optional[Path] = None, rotate_back: bool = True,
                   restricted_access_to_saved: bool = False) -> None:
        if path is None:
            path = self.path

        image_bgr = self.image_bgr

        if rotate_back and self.clockwise_rotation_index:
            image_bgr = self.rotate_image(image_bgr, -self.clockwise_rotation_index)

        cv2.imwrite(str(path), image_bgr)
        if restricted_access_to_saved:
            os.chmod(path, 0o700)

    @staticmethod
    def convert_bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def convert_rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def rotate_image(image: np.ndarray, clockwise_rotation_index: int) -> np.ndarray:
        """Rotate an image array by 90° increments clockwise. For counterclockwise rotation use negative indices."""

        if image is None:
            raise ValueError("Cannot rotate None image")

        clipped_clockwise_rotation_index = clockwise_rotation_index % 4

        if clipped_clockwise_rotation_index == 0:
            return image
        elif clipped_clockwise_rotation_index == 1:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif clipped_clockwise_rotation_index == 2:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif clipped_clockwise_rotation_index == 3:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


@dataclass
class VideoPrimitive:
    """Represents a video. Wraps path and frame-level access."""
    path: Path
    clockwise_rotation_index: int = 0
    current_frame_index: int = 0

    _video_capture: Optional[cv2.VideoCapture] = field(default=None, init=False)
    _total_frames: Optional[int] = field(default=None, init=False)
    _fps: Optional[float] = field(default=None, init=False)
    _width: Optional[int] = field(default=None, init=False)
    _height: Optional[int] = field(default=None, init=False)

    def _ensure_video_capture_is_open(self) -> cv2.VideoCapture:
        """Ensure that the video capture is opened."""
        if self._video_capture is None:
            self._video_capture = cv2.VideoCapture(str(self.path))

            if not self._video_capture.isOpened():
                raise RuntimeError(f"Failed to open video {self.path}")

        return self._video_capture

    @property
    def total_frames(self) -> int:
        if self._total_frames is None:
            self._total_frames = int(self._ensure_video_capture_is_open().get(cv2.CAP_PROP_FRAME_COUNT))
        return self._total_frames

    @property
    def fps(self) -> float:
        if self._fps is None:
            self._fps = self._ensure_video_capture_is_open().get(cv2.CAP_PROP_FPS)
        return self._fps

    @property
    def width(self) -> int:
        if self._width is None:
            self._width = int(self._ensure_video_capture_is_open().get(cv2.CAP_PROP_FRAME_WIDTH))
        return self._width

    @property
    def height(self) -> int:
        if self._height is None:
            self._height = int(self._ensure_video_capture_is_open().get(cv2.CAP_PROP_FRAME_HEIGHT))
        return self._height

    def load_frame_as_array(self, index: int) -> np.ndarray:
        """Get frame at the specified index as numpy array."""
        video_capture = self._ensure_video_capture_is_open()

        # check that index is valid
        if index < 0 or index >= self.total_frames:
            raise IndexError(f"Frame index {index} out of range <0, {self.total_frames})")

        video_capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        frame_read_successfully, frame = video_capture.read()

        if not frame_read_successfully:
            raise RuntimeError(f"Could not read frame {index}")
        return frame

    def get_frame_primitive(self, index: Optional[int] = None) -> ImagePrimitive:
        """
        Get the next frame or a frame at specific index, update current index to the index of the next frame and
        return the frame as ImagePrimitive.
        """
        if index is None:
            index = self.current_frame_index

        frame_bgr = self.load_frame_as_array(index)
        self.current_frame_index = index + 1

        return ImagePrimitive.from_not_rotated_image(frame_bgr, clockwise_rotation_index=self.clockwise_rotation_index)

    def reset_current_frame_index(self) -> None:
        """Resets the current frame index to the start (index 0)."""
        self.current_frame_index = 0

    def release(self) -> None:
        """Release the video capture handle."""
        if self._video_capture is not None:
            self._video_capture.release()
            self._video_capture = None

    def __enter__(self) -> VideoPrimitive:
        self._ensure_video_capture_is_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

    def __iter__(self):
        self.reset_current_frame_index()
        return self

    def __next__(self):
        if self.current_frame_index >= self.total_frames:
            raise StopIteration
        return self.get_frame_primitive()
