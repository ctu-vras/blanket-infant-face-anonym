from __future__ import annotations
from pathlib import Path
from typing import Type, Optional

from blanket.constants.paths import ANONYMIZER_PARAMETERS_FOLDER
from blanket.constants.enums.anonymization_enums import AnonymizationMethod
from blanket.anonymization.base_anonymizer import BaseAnonymizer
from blanket.settings.module_settings.anonymizer_settings import AnonymizerSettings
from blanket.core.config_loader import create_settings_with_extras_from_config_file

from blanket.anonymization.anonymizers.black_box import BlackBoxAnonymizer
from blanket.anonymization.anonymizers.gaussian_blur import GaussianBlurAnonymizer
from blanket.anonymization.anonymizers.pixelation import PixelationAnonymizer
from blanket.anonymization.anonymizers.stable_diffusion import StableDiffusionAnonymizer
from blanket.anonymization.anonymizers.facefusion import FacefusionAnonymizer
from blanket.anonymization.anonymizers.stable_diffusion_conditioned_facefusion import (
    StableDiffusionConditionedFacefusionAnonymizer)


anonymizer_registry: dict[AnonymizationMethod, tuple[Type[BaseAnonymizer], Path]] = {
    AnonymizationMethod.BLACK_BOX: (BlackBoxAnonymizer, Path("black_box_parameters.yaml")),
    AnonymizationMethod.GAUSSIAN_BLUR: (GaussianBlurAnonymizer, Path("gaussian_blur_parameters.yaml")),
    AnonymizationMethod.PIXELATION: (PixelationAnonymizer, Path("pixelation_parameters.yaml")),
    AnonymizationMethod.STABLE_DIFFUSION: (StableDiffusionAnonymizer, Path("sdwebui_parameters.yaml")),
    AnonymizationMethod.FACEFUSION: (FacefusionAnonymizer, Path("facefusion_parameters.yaml")),
    AnonymizationMethod.STABLE_DIFFUSION_CONDITIONED_FACEFUSION: (
        StableDiffusionConditionedFacefusionAnonymizer, Path("sdwebui_conditioned_facefusion_parameters.yaml"))
}


class AnonymizerFactory:
    @staticmethod
    def create_anonymizer(anonymization_method: AnonymizationMethod) -> BaseAnonymizer:
        """
        Create an anonymizer instance for the given module.
        Args:
            anonymization_method (AnonymizationMethod): Enum value for anonymizer type.
        Returns:
            BaseAnonymizer: Instantiated anonymizer.
        Raises:
            ValueError: If module is unknown.
        """
        anonymizer_registry_entry = anonymizer_registry.get(anonymization_method)

        if anonymizer_registry_entry is not None:
            anonymizer_class, parameters_filename = anonymizer_registry_entry
            anonymizer_parameters = create_settings_with_extras_from_config_file(
                ANONYMIZER_PARAMETERS_FOLDER / parameters_filename, AnonymizerSettings
            )
            return anonymizer_class(anonymizer_parameters)
        else:
            raise ValueError(
                f"Unknown anonymization method: {anonymization_method}. " 
                f"Available methods: {list(anonymizer_registry.keys())}"
            )


class AnonymizerCache:
    """
    Use:
        detector = AnonymizerCache.get_face_detector(FaceDetectorModule.YOLO)
    """
    _cache_instance: Optional[AnonymizerCache] = None

    def __init__(self):
        if AnonymizerCache._cache_instance is not None:
            raise RuntimeError("Trying to initialize DetectorCache multiple times")

        self._cache: dict[str, BaseAnonymizer] = {}

    @classmethod
    def get_anonymizer(cls, anonymization_method: AnonymizationMethod) -> BaseAnonymizer:
        return cls._get_cache_instance()._get_anonymizer(anonymization_method)

    def _get_anonymizer(self, anonymization_method: AnonymizationMethod) -> BaseAnonymizer:
        key = f"face:{anonymization_method.value}"
        if key not in self._cache:
            self._cache[key] = AnonymizerFactory.create_anonymizer(anonymization_method)
        return self._cache[key]

    def clear(self):
        """Optional: clear the cache if you need to reload models."""
        self._cache.clear()

    @classmethod
    def _get_cache_instance(cls):
        if cls._cache_instance is None:
            cls._cache_instance = AnonymizerCache()
        return cls._cache_instance
