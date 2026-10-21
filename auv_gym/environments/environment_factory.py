from collections.abc import Callable
import importlib
import os

import auv_gym.tools.utils as utils
from auv_gym.environments.rotation import (
    FiveMarkerSpin,
    OneMarkerSpin,
    RotationTask,
)
from auv_gym.environments.stationary import StationaryCubeTask
from auv_gym.auv_adapter import AUVAdapter
from auv_gym.robot_adapter import BoxfishAdapter, RobotAdapter
from auv_gym.vision_adapter import BoxfishVisionAdapter, VisionAdapter


RobotFactory = Callable[[object], RobotAdapter]
VisionFactory = Callable[[object], VisionAdapter]
AdapterFactory = Callable[[object, object], AUVAdapter]


class EnvironmentFactory:
    _robot_factories: dict[str, RobotFactory] = {}
    _vision_factories: dict[str, VisionFactory] = {}
    _adapter_factories: dict[str, AdapterFactory] = {}

    def __init__(self):
        pass

    @classmethod
    def register_robot_adapter(cls, domain: str, factory: RobotFactory) -> None:
        """Register a robot adapter factory for a non-Boxfish domain."""
        if not domain or domain == "boxfish":
            raise ValueError("Register custom adapters under a non-Boxfish domain")
        cls._robot_factories[domain] = factory

    @classmethod
    def register_backend(
        cls,
        domain: str,
        robot_factory: RobotFactory = None,
        vision_factory: VisionFactory = None,
        adapter_factory: AdapterFactory = None,
    ) -> None:
        """Register separate control/vision factories or one unified adapter."""
        if adapter_factory is not None:
            if robot_factory is not None or vision_factory is not None:
                raise ValueError(
                    "Pass adapter_factory or robot_factory/vision_factory, not both"
                )
            if not domain or domain == "boxfish":
                raise ValueError("Register custom adapters under a non-Boxfish domain")
            cls._adapter_factories[domain] = adapter_factory
            return
        if robot_factory is None or vision_factory is None:
            raise ValueError(
                "Provide robot_factory and vision_factory, or provide adapter_factory"
            )
        cls.register_robot_adapter(domain, robot_factory)
        cls._vision_factories[domain] = vision_factory

    @staticmethod
    def _load_adapter_plugin() -> None:
        """Load an optional adapter module before resolving a custom domain."""
        module_name = os.environ.get("AUV_GYM_ADAPTER_MODULE")
        if module_name:
            importlib.import_module(module_name)

    def create_environment(
        self,
        domain: str,
        task: str,
        *,
        env_config=None,
        auv_config=None,
        robot: RobotAdapter = None,
        vision: VisionAdapter = None,
    ):
        """
        Create an environment based on the domain and task.

        Args:
        domain: The domain of the environment.
        task: The task of the environment.

        Returns:
        Environment: The environment object.
        """

        config_dir = os.path.expanduser(
            os.environ.get("AUV_GYM_CONFIG_DIR", "~/main/configs")
        )
        self._load_adapter_plugin()

        if env_config is None:
            env_config_path = os.path.join(config_dir, "auv_env_config.json")
            env_config = utils.load_auv_env_config(env_config_path)

        if robot is None and vision is None:
            adapter_factory = self._adapter_factories.get(domain)
            if adapter_factory is not None:
                adapter = adapter_factory(env_config, auv_config)
                if not isinstance(adapter, AUVAdapter):
                    raise TypeError(
                        "adapter_factory must return an AUVAdapter instance"
                    )
                robot = adapter.robot
                vision = adapter.vision

        if robot is None:
            if domain == "boxfish":
                if auv_config is None:
                    auv_config_path = os.path.join(config_dir, "auv_config.json")
                    auv_config = utils.load_auv_config(auv_config_path)
                robot = BoxfishAdapter(auv_config)
            else:
                adapter_factory = self._robot_factories.get(domain)
                if adapter_factory is None:
                    raise ValueError(
                        f"No robot adapter registered for domain {domain!r}; "
                        "pass robot=... or call register_robot_adapter()"
                    )
                robot = adapter_factory(auv_config)

        if vision is None:
            if domain == "boxfish":
                vision = BoxfishVisionAdapter(env_config)
            else:
                vision_factory = self._vision_factories.get(domain)
                if vision_factory is None:
                    raise ValueError(
                        f"No vision adapter registered for domain {domain!r}; "
                        "pass vision=... or call register_backend()"
                    )
                vision = vision_factory(env_config)

        environment = None
        if task == "stationary":
            environment = StationaryCubeTask(
                env_config, auv_config, robot=robot, vision=vision
            )
        elif task == "rotation":
            environment = RotationTask(
                env_config, auv_config, robot=robot, vision=vision
            )
        elif task == "FiveMarkerSpin":
            environment = FiveMarkerSpin(
                env_config, auv_config, robot=robot, vision=vision
            )
        elif task == "OneMarkerSpin":
            environment = OneMarkerSpin(
                env_config, auv_config, robot=robot, vision=vision
            )

        if environment is None:
            raise ValueError(f"Invalid domain or task: {domain}, {task}")

        return environment
