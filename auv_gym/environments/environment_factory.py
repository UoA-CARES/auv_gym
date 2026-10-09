from auv_gym.environments.stationary import StationaryCubeTask
from auv_gym.environments.rotation import RotationTask, FiveMarkerSpin, OneMarkerSpin
import os
import auv_gym.tools.utils as utils
from auv_gym.robot_adapter import RobotAdapter


class EnvironmentFactory:
    def __init__(self):
        pass

    def create_environment(
        self,
        domain: str,
        task: str,
        *,
        env_config=None,
        auv_config=None,
        robot: RobotAdapter = None,
    ):
        """
        Create an environment based on the domain and task.

        Args:
        domain: The domain of the environment.
        task: The task of the environment.

        Returns:
        Environment: The environment object.
        """

        if env_config is None:
            env_config_path = os.path.expanduser("~/main/configs/auv_env_config.json")
            env_config = utils.load_auv_env_config(env_config_path)

        if robot is None and auv_config is None:
            if domain != "boxfish":
                raise ValueError("A non-Boxfish domain requires a RobotAdapter")
            auv_config_path = os.path.expanduser("~/main/configs/auv_config.json")
            auv_config = utils.load_auv_config(auv_config_path)

        environment = None
        if domain == "boxfish" or robot is not None:
            if task == "stationary":
                environment = StationaryCubeTask(env_config, auv_config, robot=robot)
            elif task == "rotation":
                environment = RotationTask(env_config, auv_config, robot=robot)
            elif task == "FiveMarkerSpin":
                environment = FiveMarkerSpin(env_config, auv_config, robot=robot)
            elif task == "OneMarkerSpin":
                environment = OneMarkerSpin(env_config, auv_config, robot=robot)

        if environment is None:
            raise ValueError(f"Invalid domain or task: {domain}, {task}")

        return environment
