from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from blanket.core.geometry import SO3


@dataclass
class FaceDetection:
    """Stores a face bounding box (axis aligned) and optionally associated facial landmarks."""

    left_top_right_bottom: np.ndarray  # [left, top, right, bottom], int32
    confidence: Optional[float] = None
    facial_landmarks_detection: Optional[FacialLandmarksDetection] = None

    def __post_init__(self):
        """
        Round bounding box coordinates to int32 after initialization.
        """
        self.left_top_right_bottom = np.round(self.left_top_right_bottom).astype(np.int32)

    @staticmethod
    def from_left_top_width_height(
        left_top_width_height: np.ndarray, confidence: Optional[int] = None
    ) -> FaceDetection:
        """
        Create FaceDetection from left, top, width, height format.
        Args:
            left_top_width_height (np.ndarray): [left, top, width, height]
            confidence (Optional[int]): Detection confidence
        Returns:
            FaceDetection: Detection object
        """
        return FaceDetection(
            left_top_width_height + np.asarray([0, 0, left_top_width_height[0], left_top_width_height[1]]), confidence
        )

    @property
    def left_top_width_height(self) -> np.ndarray:
        """
        Get bounding box in left, top, width, height format.
        Returns:
            np.ndarray: [left, top, width, height]
        """
        return self.left_top_right_bottom - np.asarray([0, 0, self.left, self.top])

    @property
    def left(self) -> int:
        """Get left coordinate of bounding box."""
        return int(self.left_top_right_bottom[0])

    @property
    def top(self) -> int:
        """Get top coordinate of bounding box."""
        return int(self.left_top_right_bottom[1])

    @property
    def right(self) -> int:
        """Get right coordinate of bounding box."""
        return int(self.left_top_right_bottom[2])

    @property
    def bottom(self) -> int:
        """Get bottom coordinate of bounding box."""
        return int(self.left_top_right_bottom[3])

    @property
    def width(self) -> int:
        """Get width of bounding box."""
        return int(self.left_top_right_bottom[2] - self.left_top_right_bottom[0])

    @property
    def height(self) -> int:
        """Get height of bounding box."""
        return int(self.left_top_right_bottom[3] - self.left_top_right_bottom[1])

    @property
    def area(self) -> int:
        """
        Compute the area of the bounding box.
        Returns:
            int: Area in pixels
        """
        width, height = self.left_top_right_bottom[2:] - self.left_top_right_bottom[:2]
        return int(width * height)

    @property
    def center(self) -> np.ndarray:
        """
        Get center coordinates of bounding box.
        Returns:
            np.ndarray: [x, y] center
        """
        return (self.left_top_right_bottom[:2] + self.left_top_right_bottom[2:]) // 2

    @staticmethod
    def intersection_over_union(first_detection: FaceDetection, second_detection: FaceDetection) -> float:
        """
        Compute Intersection over Union (IoU) of two bounding boxes.
        Args:
            first_detection (FaceDetection): First bounding box
            second_detection (FaceDetection): Second bounding box
        Returns:
            float: IoU value
        """
        intersection_left_top = np.maximum(
            first_detection.left_top_right_bottom[:2], second_detection.left_top_right_bottom[:2]
        )
        intersection_right_bottom = np.minimum(
            first_detection.left_top_right_bottom[2:], second_detection.left_top_right_bottom[2:]
        )

        intersection_width_height = np.maximum(0, intersection_right_bottom - intersection_left_top)  # (width, height)
        intersection_area = intersection_width_height[0] * intersection_width_height[1]

        union_area = first_detection.area + second_detection.area - intersection_area

        return float(intersection_area / union_area) if union_area > 0 else 0.0

    @staticmethod
    def center_distance(first_detection: FaceDetection, second_detection: FaceDetection) -> float:
        """
        Compute Euclidean distance between centers of two bounding boxes.
        Args:
            first_detection (FaceDetection): First bounding box
            second_detection (FaceDetection): Second bounding box
        Returns:
            float: Distance in pixels
        """
        return float(np.linalg.norm(first_detection.center - second_detection.center))

    def create_mask(self, image_shape: tuple[int, int, int]) -> np.ndarray:
        """
        Create binary mask for the face using landmarks.
        Args:
            image_shape (tuple): Shape of the image (H, W, C)
        Returns:
            np.ndarray: Binary mask
        """
        if self.facial_landmarks_detection is None:
            raise ValueError("Cannot create mask without landmarks")
        return self.facial_landmarks_detection.convex_hull_binary_mask(image_shape)

    def __str__(self):
        """String representation of FaceDetection."""
        return f"FaceDetection(ltrb={self.left_top_right_bottom}, confidence={self.confidence})"


# could turn Face detection into general abstract class and add two subclasses for axis aligned bounding boxes and
#  for bounding boxes that can be rotated


@dataclass
class FacialLandmarksDetection:
    """Stores pixel coordinates of facial landmarks and optionally 3D facial orientation and confidence."""

    landmarks: np.ndarray  # shape (num_landmarks, 2)
    orientation: Optional[SO3] = None
    confidence: Optional[np.ndarray] = None  # per-landmark confidence

    _mask: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def get_specific_landmarks(self, landmark_indices: list[int]) -> np.ndarray:
        """
        Return selected landmark coordinates.
        Args:
            landmark_indices (list[int]): Indices of landmarks to select
        Returns:
            np.ndarray: Selected landmark coordinates
        """
        return self.landmarks[landmark_indices]

    def mean_point(self, landmark_indices: Optional[list[int]] = None) -> np.ndarray:
        """
        Compute mean point (arithmetic) of all or selected landmarks.
        Args:
            landmark_indices (Optional[list[int]]): Indices of landmarks to use
        Returns:
            np.ndarray: Mean point coordinates
        """
        if landmark_indices is None:
            landmarks = self.landmarks
        else:
            landmarks = self.landmarks[landmark_indices]

        return landmarks.mean(axis=0)

    def convex_hull_binary_mask(self, image_shape: tuple[int, int, int], force_recompute=False) -> np.ndarray:
        """
        Return a binary mask of the landmarks convex hull (or its cached value).
        Args:
            image_shape (tuple): Shape of the image (H, W, C)
            force_recompute (bool): If True, recompute mask even if cached
        Returns:
            np.ndarray: Binary mask
        """
        if not force_recompute and self._mask is not None and self._mask.shape == image_shape:
            return self._mask

        self._mask = np.zeros(image_shape, np.uint8)
        hull = cv2.convexHull(self.landmarks)
        cv2.fillConvexPoly(self._mask, hull, color=(255, 255, 255))

        return self._mask
