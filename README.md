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

The current implementation is validated with a Boxfish AUV. The Boxfish SDK and `boxfish_lib` are private integration dependencies and are not distributed with this repository. Boxfish is one vehicle implementation, not a requirement of the testbed design: compatible vehicles can be connected through `AUVAdapter` and the adapter interfaces.

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
- [Hardware setup](#hardware-setup)
- [Getting started](#getting-started)
- [Example configuration](#example-configuration)
- [Run the Gymnasium training path](#run-the-gymnasium-training-path)
- [Use with your AUV](#use-with-your-auv)
- [Tasks and observations](#tasks-and-observations)
- [Repository layout](#repository-layout)
- [License and project status](#license-and-project-status)

## Requirements

The supported training path requires:

- Python 3.10.
- [CARES reinforcement learning (`feat/auv_gym`)](https://github.com/UoA-CARES/cares_reinforcement_learning/tree/feat/auv_gym).
- [Gymnasium environments (`feat/auv_gym`)](https://github.com/UoA-CARES/gymnasium_envrionments/tree/feat/auv_gym).
- A camera, calibration files, fiducial-marker setup, and an AUV control/SDK package.
- An [`AUVAdapter`](auv_gym/auv_adapter.py) implementation for your vehicle.


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

python -m pip install -r requirements.txt
python -m pip install -r ../cares_reinforcement_learning/requirements.txt
python -m pip install -r ../gymnasium_envrionments/requirements.txt
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

Install your AUV SDK in the same virtual environment. The private Boxfish SDK and `boxfish_lib` are not included here; authorized Boxfish users must provide them separately when using the built-in `boxfish` backend. Other users should provide their own SDK and adapter.

Before running a physical experiment, verify the installed revisions, camera calibration, marker IDs, vehicle action limits, and stop behavior.

## Example configuration

The [`configs/`](configs) directory is a template showing the expected file layout. It contains placeholder values and calibration files; replace them before running a vehicle.

For a first setup, copy it to a separate working directory and edit:

- `env_config.json` for the Gymnasium domain and task.
- `auv_env_config.json` for camera, marker, and episode settings.
- `my_auv_config.json` for your native AUV settings.
- `alg_config.json` and `train_config.json` for the experiment.

Use the copied directory as both `--data_path` and `AUV_GYM_CONFIG_DIR` if your adapter follows the example layout.

## Run the Gymnasium training path

The supported entry point is the `run.py` inside `gymnasium_envrionments/scripts`. AUV Gym supplies the environment and adapter code; the Gymnasium repository supplies the training entry point:

```bash
python3 /path/to/gymnasium_envrionments/scripts/run.py train config \
  --data_path /path/to/training-configs
```

The `--data_path` directory must contain `alg_config.json`, `env_config.json`, and `train_config.json`. The command starts the CARES RL runner, which creates the Gymnasium AUV wrapper and then calls AUV Gym's `EnvironmentFactory`.

This is an online hardware path, not a simulator smoke test. Resetting or stepping can access the camera and issue movement commands.

## Use with your AUV

The integration has two configuration layers:

1. The Gymnasium runner reads `alg_config.json`, `env_config.json`, and
   `train_config.json` from `--data_path`.
2. AUV Gym reads `auv_env_config.json` from `AUV_GYM_CONFIG_DIR`. This contains
   camera, calibration, marker, display, and episode settings.

For a custom vehicle, the adapter plugin loads any additional native vehicle configuration it needs. The private Boxfish backend additionally reads `auv_config.json` from `AUV_GYM_CONFIG_DIR`.

### 1. Implement the adapter

Create a Python module for your vehicle. It must register a domain and return an `AUVAdapter`:

```python
# my_auv_backend.py
import json
import os
from pathlib import Path

from auv_gym import AUVAdapter
from auv_gym.environments.environment_factory import EnvironmentFactory

from my_auv_adapters import MyRobotAdapter, MyVisionAdapter


def make_adapter(env_config, _auv_config):
    config_path = Path(os.environ["MY_AUV_CONFIG"])
    with config_path.open(encoding="utf-8") as file:
        auv_config = json.load(file)
    return AUVAdapter(
        MyRobotAdapter(auv_config),
        MyVisionAdapter(env_config, auv_config),
    )


EnvironmentFactory.register_backend(
    "my-auv",
    adapter_factory=make_adapter,
)
```

Make the module importable by installing its package or adding its directory to `PYTHONPATH`. The runner loads it before creating the environment:

```bash
export PYTHONPATH=/path/to/my-auv-adapter:$PYTHONPATH
export AUV_GYM_ADAPTER_MODULE=my_auv_backend
```

### 2. Implement the control and vision contracts

`MyRobotAdapter` must implement:

- `control_actions`, `min_values`, `max_values`, `gripper_marker_ids`, and
  `action_type`.
- `info()`, `safety_check()`, `home()`, `move(actions)`, and `stop()`.
- Task-specific methods such as `level()`, `reverse_hard_stop()`, or
  `move_roll_grabber()` when the selected task requires them.
- `close()` to stop the vehicle and release SDK resources.

`MyVisionAdapter` must provide:

- `camera`, with `get_frame()`, `camera_matrix`, and `camera_distortion`.
- `create_marker_detector(...)`, returning a detector with
  `get_marker_poses(...)`.

Each marker pose must contain at least `position`; rotation tasks also use `orientation`, and the stationary task may use `r_vec`. `RobotAdapter.info()` must return the robot state expected by the selected task.

Both interfaces may wrap the same native AUV client. They do not require two physical devices or two SDK connections.

### 3. Configure the environment and task

Copy the training configs and change the domain from `boxfish` to your adapter's domain:

```json
{
  "gym": "auv",
  "domain": "my-auv",
  "task": "FiveMarkerSpin",
  "save_train_checkpoints": 1
}
```

Create an AUV Gym configuration directory containing at least:

```text
/path/to/my-auv-config/
├── auv_env_config.json
├── camera_matrix.txt
└── camera_distortion.txt
```

Set the paths in `auv_env_config.json` to the calibration files and point the factory at that directory:

```bash
export AUV_GYM_CONFIG_DIR=/path/to/my-auv-config
export MY_AUV_CONFIG=/path/to/my-auv-config/my_auv_config.json
```

### 4. Run training

```bash
python3 /path/to/gymnasium_envrionments/scripts/run.py train config \
  --data_path /path/to/training-configs
```

The CARES RL and Gymnasium repositories do not need to be modified. The AUV Gym wrapper preserves their existing runner contract while the adapter handles vehicle control, camera access, and marker detection.

## Tasks and observations

The factory includes a private `boxfish` domain and custom domains registered by adapter plugins. The available task names are:

| Task | Implementation | Purpose |
| --- | --- | --- |
| `stationary` | `StationaryCubeTask` | Move a marked cube relative to the gripper markers. |
| `rotation` | `RotationTask` | Abstract base for rotation tasks. |
| `OneMarkerSpin` | `OneMarkerSpin` | Spin toward a selected inner-wheel marker target. |
| `FiveMarkerSpin` | `FiveMarkerSpin` | Spin using a randomly selected inner-wheel marker target. |

`RotationTask` is abstract in the current source, so use `OneMarkerSpin` or `FiveMarkerSpin` for a concrete rotation environment.

The built-in tasks return NumPy observations assembled from robot state and marker poses. Their exact size and meaning depend on the task. The current validated experiments use `OneMarkerSpin` and `FiveMarkerSpin`.

## Configuration reference

When using the private `boxfish` domain, `AUV_GYM_CONFIG_DIR` must contain:

```text
~/main/configs/auv_env_config.json
~/main/configs/auv_config.json
```

`~/main/configs` is only the default. Set `AUV_GYM_CONFIG_DIR` explicitly for your vehicle so the runner does not depend on that path.

`auv_env_config.json` contains camera, calibration, marker-size, display, tolerance, and episode settings. For the private Boxfish integration, `auv_config.json` contains the control method, device name, action names, action bounds, home sequence, and marker IDs. Custom adapters should load their own native configuration in the plugin.

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
│   ├── auv_adapter.py
│   ├── robot_adapter.py
│   └── vision_adapter.py
├── tests/
│   └── test_backend_bridge.py
├── LICENSE
├── LICENSE-HARDWARE
└── setup.py
```

The former direct-runner/trainer files are not part of the supported
Gymnasium training path. Do not use a second AUV Gym training entry point; load
your adapter through `AUV_GYM_ADAPTER_MODULE` and start the Gymnasium runner.


## License and project status

Research use of this testbed is welcomed by the project maintainers.

- The Python software in this repository is licensed under the [MIT License](LICENSE).
- The CAD and hardware source files in [`auv_gym/hardware`](auv_gym/hardware) are licensed under [CERN-OHL-P-2.0](LICENSE-HARDWARE).
- The photographs, pool video, and other media are not automatically covered by either license. Their reuse should be cleared separately with the rights holders.
- CARES RL, camera software, and other third-party dependencies remain under their own licenses.

The copyright notices use the authors listed in the accompanying paper. Confirm authorship, institutional approval, and any third-party rights before publishing or redistributing the repository.
