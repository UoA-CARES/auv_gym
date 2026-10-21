from auv_gym.auv_adapter import AUVAdapter
from auv_gym.environments.environment_factory import EnvironmentFactory
from auv_gym.robot_adapter import RobotAdapter
from auv_gym.tools.configurations import AUVEnvironmentConfig
from auv_gym.vision_adapter import VisionAdapter


class FakeRobot(RobotAdapter):
    def __init__(self):
        super().__init__(
            control_actions=["roll"],
            min_values=[-1],
            max_values=[1],
            gripper_marker_ids=[18, 19],
        )
        self.home_called = False
        self.safety_called = False
        self.closed = False

    def info(self):
        return None

    def safety_check(self):
        self.safety_called = True

    def home(self):
        self.home_called = True

    def move(self, actions):
        pass

    def stop(self):
        pass

    def close(self):
        self.closed = True

    def spin_by_angle(self, angle):
        pass


class FakeVision(VisionAdapter):
    class Camera:
        def release(self):
            pass

    def __init__(self):
        self._camera = self.Camera()
        self.detector_created = False

    @property
    def camera(self):
        return self._camera

    def create_marker_detector(self, marker_size, library_hd, error_correction=-1):
        self.detector_created = True
        return object()


def test_custom_backend_can_create_task_without_boxfish():
    robot = FakeRobot()
    vision = FakeVision()
    EnvironmentFactory.register_backend(
        "fake-auv",
        lambda config: robot,
        lambda config: vision,
    )

    config = AUVEnvironmentConfig(
        camera_id=0,
        camera_matrix="camera.txt",
        camera_distortion="distortion.txt",
        display=False,
    )
    environment = EnvironmentFactory().create_environment(
        "fake-auv",
        "OneMarkerSpin",
        env_config=config,
    )

    assert environment.robot is robot
    assert robot.home_called
    assert robot.safety_called
    assert vision.detector_created

    environment.close()
    assert robot.closed


def test_unified_adapter_can_create_task_without_boxfish():
    robot = FakeRobot()
    vision = FakeVision()
    EnvironmentFactory.register_backend(
        "unified-fake-auv",
        adapter_factory=lambda env_config, auv_config: AUVAdapter(robot, vision),
    )

    config = AUVEnvironmentConfig(
        camera_id=0,
        camera_matrix="camera.txt",
        camera_distortion="distortion.txt",
        display=False,
    )
    environment = EnvironmentFactory().create_environment(
        "unified-fake-auv",
        "OneMarkerSpin",
        env_config=config,
    )

    assert environment.adapter.robot is robot
    assert environment.adapter.vision is vision
    environment.close()
    assert robot.closed
