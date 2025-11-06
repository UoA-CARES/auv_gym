from auv_gym.environments.stationary import StationaryCubeTask
from auv_gym.environments.rotation import RotationTask, JustSpin
import os
import auv_gym.tools.utils as utils


class EnvironmentFactory:
    def __init__(self):
        pass

    def create_environment(self, domain: str, task: str):
        """
        Create an environment based on the domain and task.

        Args:
        domain: The domain of the environment.
        task: The task of the environment.

        Returns:
        Environment: The environment object.
        """

        env_config_path = os.path.expanduser(f"~/main/configs/auv_env_config.json")
        env_config = utils.load_auv_env_config(env_config_path)

        auv_config_path = os.path.expanduser(f"~/main/configs/auv_config.json")
        auv_config = utils.load_auv_config(auv_config_path)

        environment = None
        if domain == "boxfish":
            if task == "stationary":
                environment = StationaryCubeTask(env_config, auv_config)
            elif task == "rotation":
                environment = RotationTask(env_config, auv_config)
            elif task == "just_spin":
                environment = JustSpin(env_config, auv_config)

        if environment is None:
            raise ValueError(f"Invalid domain or task: {domain}, {task}")

        return environment
