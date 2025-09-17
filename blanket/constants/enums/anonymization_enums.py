from enum import Enum


class AnonymizationMethod(str, Enum):
    BLACK_BOX = "black_box"
    GAUSSIAN_BLUR = "gaussian_blur"
    PIXELATION = "pixelation"
    STABLE_DIFFUSION = "stable_diffusion"
    FACEFUSION = "facefusion"
    STABLE_DIFFUSION_CONDITIONED_FACEFUSION = "stable_diffusion_conditioned_facefusion"


class MatchingMethod(str, Enum):
    IOU = "intersection_over_union"
    DISTANCE = "distance"
    CONFIDENCE = "confidence"


class PaddingMethod(str, Enum):
    RATIO = "ratio"
    CONSTANT = "constant"
