<h1 align="center">AUV Gym</h1>


<table>
<tr>
<td width="30%" valign="top">

<a href="auv_gym/image/IMG_0560.MOV">
  <img src="auv_gym/image/IMG_0560_preview.gif" alt="Animated preview of the underwater testbed operating in the pool" width="100%">
</a>
<p><small><a href="auv_gym/image/IMG_0560.MOV">Open the full-resolution pool video</a></small></p>

</td>
<td width="70%" valign="top">

AUV Gym is the software component of an open-source, robot-agnostic underwater manipulation testbed. It connects visual marker detection, vehicle control, task-specific rewards, and the <a href="https://github.com/UoA-CARES/cares_reinforcement_learning/tree/feat/auv_gym">CARES reinforcement-learning library</a>.

The current implementation is validated with a Boxfish AUV. Boxfish is one vehicle implementation, not a requirement of the testbed design: compatible vehicles can be connected through `AUVAdapter` and the adapter interfaces.

> **Hardware warning:** this repository is intended for experiments with a physical vehicle and camera. Resetting or stepping an environment can access the camera and issue movement commands. Test with the vehicle secured, confirm action limits, and keep an operator able to stop the vehicle.

[![Python baseline](https://img.shields.io/badge/python%20baseline-3.10-blue.svg)](#requirements)
[![Software license](https://img.shields.io/badge/software%20license-MIT-green.svg)](LICENSE)
[![Hardware license](https://img.shields.io/badge/hardware%20license-CERN--OHL--P--2.0-orange.svg)](LICENSE-HARDWARE)

</td>
</tr>
</table>

The looping GIF preview is used here because GitHub does not reliably render repository-local `<video>` tags in README pages. Click the preview to open the full `.MOV` video.



## Contents

- [Requirements](#requirements)
- [Getting started](#getting-started)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Hardware setup](#hardware-setup)
- [Robot adapters](#robot-adapters)
- [Tasks and observations](#tasks-and-observations)
- [Repository layout](#repository-layout)
- [License and project status](#license-and-project-status)

## Requirements

The factory-based environment path requires:

- [CARES reinforcement learning (`feat/auv_gym`)](https://github.com/UoA-CARES/cares_reinforcement_learning/tree/feat/auv_gym).
- [Gymnasium environments (`feat/auv_gym`)](https://github.com/UoA-CARES/gymnasium_envrionments/tree/feat/auv_gym).
- A compatible camera, calibration files, fiducial-marker setup, and vehicle-control configuration.

Also install your vehicle's control/SDK package and provide the adapter described in [Robot adapters](#robot-adapters). 

## Getting started

Clone the repositories beside one another, create a virtual environment, and install the local packages in editable mode:

```bash
git clone https://github.com/UoA-CARES/auv_gym.git
git clone --branch feat/auv_gym https://github.com/UoA-CARES/cares_reinforcement_learning.git
git clone --branch feat/auv_gym https://github.com/UoA-CARES/gymnasium_envrionments.git

cd auv_gym
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

python -m pip install -r ../cares_reinforcement_learning/requirements.txt
python -m pip install -e ../cares_reinforcement_learning
python -m pip install -e .
```

The relative paths assume this layout:

```text
parent-directory/
├── auv_gym/
├── cares_reinforcement_learning/
└── gymnasium_envrionments/
```

This repository is developed against the CARES RL and Gymnasium environments **`feat/auv_gym`** branches. Keep both checkouts on those branches; other revisions may not provide the interfaces used by AUV Gym.

Before running a physical experiment, verify the installed CARES RL revision, camera calibration, marker IDs, vehicle action limits, and stop behavior.

## Quick start

For the validated Boxfish hardware setup, use the legacy CLI below after
installing `boxfish_lib`:

```bash
python3 run.py train config --data_path /home/anyone/main/configs
```

This is an online hardware example, not a simulator smoke test. Depending on the task, marker loss can pause for operator input or truncate an episode.

For another AUV, register its adapter and run the existing CARES/Gymnasium
runner as described in [CARES RL bridge](#cares-rl-bridge).

## Usage

### Environment API

AUV Gym uses a small custom API:

- `reset()` returns the initial state.
- `step(action)` returns `state, reward, done, truncated, environment_info`.
- `sample_action()` samples an action using the configured action bounds.
- `normalize()` and `denormalize()` convert between vehicle action ranges and `[-1, 1]`.

The environment factory includes the `boxfish` domain and supports additional
domains registered by an adapter plugin. The task names are:

| Task | Implementation | Purpose |
| --- | --- | --- |
| `stationary` | `StationaryCubeTask` | Move a marked cube relative to the gripper markers. |
| `rotation` | `RotationTask` | Abstract base for rotation tasks. |
| `OneMarkerSpin` | `OneMarkerSpin` | Spin toward a selected inner-wheel marker target. |
| `FiveMarkerSpin` | `FiveMarkerSpin` | Spin using a randomly selected inner-wheel marker target. |

`RotationTask` is abstract in the current source, so use `OneMarkerSpin` or `FiveMarkerSpin` for a concrete rotation environment.

### Configuration

When the built-in `boxfish` domain uses default configuration loading,
`EnvironmentFactory` expects:

```text
~/main/configs/auv_env_config.json
~/main/configs/auv_config.json
```

`auv_env_config.json` contains camera, calibration, marker-size, display, tolerance, and episode settings. `auv_config.json` contains the vehicle control method, device name, action names, action bounds, home sequence, and gripper-marker IDs. A custom domain can use its own vehicle configuration and pass it to its adapter factory.

Example shape for `auv_env_config.json`:

```json
{
  "camera_id": 0,
  "camera_matrix": "/path/to/camera_matrix.txt",
  "camera_distortion": "/path/to/camera_distortion.txt",
  "episode_horizon": 20,
  "noise_tolerance": 15,
  "display": true
}
```

Example shape for `auv_config.json`:

```json
{
  "device_name": "boxfish",
  "control_method": "onboard",
  "baudrate": 115200,
  "control_actions": ["roll"],
  "min_values": [-90],
  "max_values": [90],
  "gripper_marker_ids": [7, 8]
}
```

These snippets are illustrative. Use values calibrated for the vehicle, camera, marker layout, and selected task.

## Hardware setup

The physical fixture uses a marked outer housing, an  inner ring, and removable toolheads for different manipulation tasks.

![Fixture, outer housing, inner ring, and toolheads](auv_gym/image/fixture_and_toolheads.jpeg)

The fabrication files are in [`auv_gym/hardware`](auv_gym/hardware):

- [`inner_ring.3mf`](auv_gym/hardware/inner_ring.3mf) is the inner ring.
- [`Outter_housing.3mf`](auv_gym/hardware/Outter_housing.3mf) is the outer housing. The 3MF files support multi-colour printing or separate prints.
- [`toolhead1/`](auv_gym/hardware/toolhead1) contains the Toolhead 1 base and its two-piece left/right mold.
- [`toolhead2/`](auv_gym/hardware/toolhead2) contains the Toolhead 2 base and its two-piece left/right mold.
- [`toolhead3.stl`](auv_gym/hardware/toolhead3.stl) is the standalone Toolhead 3 mesh.

The Toolhead 1 and Toolhead 2 mold designs were used with [Smooth-On PMC-770](https://www.smooth-on.com/products/pmc-770/). Select the printing or casting workflow appropriate for the material and fabrication equipment available for your build.

## Robot adapters

The environments depend on small adapters rather than directly on the Boxfish controller. [`auv_adapter.py`](auv_gym/auv_adapter.py) provides the public `AUVAdapter` facade. [`robot_adapter.py`](auv_gym/robot_adapter.py) provides `RobotAdapter` and includes `BoxfishAdapter`, which is used for the existing Boxfish configuration. [`vision_adapter.py`](auv_gym/vision_adapter.py) provides the camera and marker-detector contract.

To connect another robot, create a class that inherits from `RobotAdapter` and store the robot's native controller or SDK client inside it. The adapter translates AUV Gym actions into the units and commands required by the robot.

An adapter should:

1. Define `control_actions`, `min_values`, and `max_values` with matching lengths, then set the marker IDs and action type required by the task.
2. Implement `info()`, `safety_check()`, `home()`, `move(actions)`, and `stop()`.
3. Add task-specific motion methods such as `move_roll_grabber()` or `spin_by_angle()` when the selected task requires them.
4. Return the observation fields expected by the task, including the IMU and marker-pose data used by its state conversion.
5. Release connections and leave the robot stopped in `close()`.

Pass a custom `AUVAdapter` as `adapter_factory=...` when registering a domain. The facade contains a `RobotAdapter` and `VisionAdapter`, but this does not
require two physical devices or two SDK clients: if one AUV library owns both
control and camera access, both interfaces can wrap the same underlying client.

`RobotAdapter` must translate actions and expose the task's robot state.  `VisionAdapter` must expose `camera` and `create_marker_detector(marker_size, library_hd, error_correction=-1)`. The camera and detector must provide the methods and marker-pose structures used by the selected task.

### CARES RL bridge

The existing CARES/Gymnasium runner can use AUV Gym without changes to CARES RL or Gymnasium. AUV Gym preserves the runner's environment contract while keeping vehicle control and vision behind adapters.

For a reusable vehicle backend, register one unified adapter under a domain:

```python
from auv_gym import AUVAdapter
from auv_gym.environments.environment_factory import EnvironmentFactory

EnvironmentFactory.register_backend(
    "my-auv",
    adapter_factory=lambda env_config, auv_config: AUVAdapter(
        MyRobotAdapter(auv_config),
        MyVisionAdapter(env_config),
    ),
)
```

The lower-level `robot_factory=...` and `vision_factory=...` registration form remains available when the two interfaces are constructed separately.

When the existing runner is started in another process, set `AUV_GYM_ADAPTER_MODULE` to a module that performs this registration. `AUV_GYM_CONFIG_DIR` can point the default factory at that vehicle's configuration directory.


## Tasks and observations

- Observations are NumPy arrays assembled from IMU and detected-marker pose information. Their size and meaning depend on the selected task.
- `environments/reward_config.py` contains the active rotation/yaw reward implementation, including angular-distance progress, distance-to-goal shaping, and goal bonuses.
- The paper's reported experiments use the single-marker and five-marker rotation tasks. Other task code should be treated as additional or experimental functionality unless separately validated.

## Repository layout

```text
auv_gym/
├── auv_gym/
│   ├── environments/
│   │   ├── environment.py
│   │   ├── environment_factory.py
│   │   ├── rotation.py
│   │   ├── stationary.py
│   │   └── reward_config.py
│   ├── hardware/
│   │   ├── inner_ring.3mf
│   │   ├── Outter_housing.3mf
│   │   ├── toolhead1/
│   │   ├── toolhead2/
│   │   └── toolhead3.stl
│   ├── image/
│   │   ├── fixture_and_toolheads.jpeg
│   │   ├── IMG_0560.MOV
│   │   └── IMG_0560_preview.gif
│   ├── tools/
│   ├── robot_adapter.py
│   ├── auv_trainer.py
│   └── run.py
├── LICENSE
├── LICENSE-HARDWARE
├── paper.tex
└── setup.py
```


## License and project status

Research use of this testbed is welcomed by the project maintainers.

- The Python software in this repository is licensed under the [MIT License](LICENSE).
- The CAD and hardware source files in [`auv_gym/hardware`](auv_gym/hardware) are licensed under [CERN-OHL-P-2.0](LICENSE-HARDWARE).
- The photographs, pool video, and other media are not automatically covered by either license. Their reuse should be cleared separately with the rights holders.
- CARES RL, camera software, and other third-party dependencies remain under their own licenses.

The copyright notices use the authors listed in the accompanying paper. Confirm authorship, institutional approval, and any third-party rights before publishing or redistributing the repository.
