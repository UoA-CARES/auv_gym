"""Robot control adapters used by AUV Gym environments."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence


class RobotAdapter(ABC):
    """Minimal control contract required by the AUV Gym environments.

    Implement this class for a robot other than Boxfish. The pose returned by
    ``info()`` must keep the structure expected by the selected task.
    """

    def __init__(
        self,
        control_actions: Sequence[str],
        min_values: Sequence[float],
        max_values: Sequence[float],
        gripper_marker_ids: Sequence[int] = (),
        action_type: Optional[str] = None,
    ) -> None:
        if not (
            len(control_actions) == len(min_values) == len(max_values)
        ):
            raise ValueError(
                "control_actions, min_values, and max_values must have the same length"
            )

        self.control_actions = tuple(control_actions)
        self.min_values = tuple(min_values)
        self.max_values = tuple(max_values)
        self.gripper_marker_ids = tuple(gripper_marker_ids)
        self.action_type = action_type

    @abstractmethod
    def info(self) -> Any:
        """Return the robot state used by the selected environment task."""

    @abstractmethod
    def safety_check(self) -> None:
        """Check that the robot is safe to control."""

    @abstractmethod
    def home(self) -> None:
        """Move the robot to its task start state."""

    @abstractmethod
    def move(self, actions: Sequence[float]) -> None:
        """Apply the task's translational or general action vector."""

    @abstractmethod
    def stop(self) -> None:
        """Stop all robot motion."""

    def level(self) -> None:
        """Level the robot when supported; stopping is the safe fallback."""
        self.stop()

    def reverse_hard_stop(self) -> None:
        """Perform the task-specific reverse stop; stopping is the fallback."""
        self.stop()

    def move_roll_grabber(self, actions: Sequence[float]) -> None:
        """Apply a rotation/grabber action vector.

        Rotation tasks require adapters to implement this method.
        """
        raise NotImplementedError("This adapter does not support roll/grabber actions")

    def spin_by_angle(self, angle: Sequence[float]) -> None:
        """Apply an angle-based spin action."""
        raise NotImplementedError("This adapter does not support angle-based spinning")

    def close(self) -> None:
        """Release robot resources; stopping is the safe default."""
        self.stop()


class BoxfishAdapter(RobotAdapter):
    """Adapter that exposes ``boxfish_lib.Boxfish`` through ``RobotAdapter``."""

    def __init__(self, config: Any) -> None:
        # Lazy import keeps the adapter contract usable by non-Boxfish callers.
        from boxfish_lib.Boxfish import Boxfish

        self.backend = Boxfish(config)
        super().__init__(
            control_actions=config.control_actions,
            min_values=config.min_values,
            max_values=config.max_values,
            gripper_marker_ids=config.gripper_marker_ids,
            action_type=config.action_type,
        )

    def info(self) -> Any:
        return self.backend.info()

    def safety_check(self) -> None:
        self.backend.safety_check()

    def home(self) -> None:
        self.backend.home()

    def move(self, actions: Sequence[float]) -> None:
        self.backend.move(actions)

    def stop(self) -> None:
        self.backend.stop()

    def level(self) -> None:
        self.backend.level()

    def reverse_hard_stop(self) -> None:
        self.backend.reverse_hard_stop()

    def move_roll_grabber(self, actions: Sequence[float]) -> None:
        self.backend.move_roll_grabber(actions)

    def spin_by_angle(self, angle: Sequence[float]) -> None:
        self.backend.spin_by_angle(angle)

    def close(self) -> None:
        self.backend.close()
