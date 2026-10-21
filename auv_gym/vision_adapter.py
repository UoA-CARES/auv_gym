"""Vision and camera adapters used by AUV Gym tasks.

The task code only needs a camera and a marker-detector factory.  Vehicle
libraries can provide their own implementation without importing Boxfish.
"""

from abc import ABC, abstractmethod
from typing import Any


class VisionAdapter(ABC):
    """Minimal camera/marker contract required by the built-in tasks."""

    @property
    @abstractmethod
    def camera(self) -> Any:
        """Return an object with the camera methods used by the tasks."""

    @abstractmethod
    def create_marker_detector(
        self, marker_size: float, library_hd: int, error_correction: int = -1
    ) -> Any:
        """Create a detector for the selected marker family."""

    def close(self) -> None:
        """Release camera resources when the backend supports it."""
        camera = self.camera
        release = getattr(camera, "release", None)
        if callable(release):
            release()


class BoxfishVisionAdapter(VisionAdapter):
    """Boxfish camera/STag implementation kept behind the vision boundary."""

    def __init__(self, config: Any) -> None:
        # Keep Boxfish imports lazy so custom AUV integrations do not need the
        # Boxfish package merely to import AUV Gym.
        from boxfish_lib.vision.camera import Camera

        self._camera = Camera(
            config.camera_id,
            config.camera_matrix,
            config.camera_distortion,
        )

    @property
    def camera(self) -> Any:
        return self._camera

    def create_marker_detector(
        self, marker_size: float, library_hd: int, error_correction: int = -1
    ) -> Any:
        from boxfish_lib.vision.STagDetector import STagDetector

        return STagDetector(
            marker_size=marker_size,
            library_hd=library_hd,
            error_correction=error_correction,
        )
