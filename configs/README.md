# Example configuration

This directory shows the files used by the Gymnasium runner and AUV Gym. It is
a template, not a hardware-ready configuration.

Replace the placeholder AUV name, action limits, marker IDs, device settings, calibration paths, and native SDK settings before using a real vehicle.

The Gymnasium runner reads:

- `alg_config.json`
- `env_config.json`
- `train_config.json`

Set `--data_path` to this directory. AUV Gym reads `auv_env_config.json` from
the directory named by `AUV_GYM_CONFIG_DIR`. The example adapter in the main
README reads `my_auv_config.json` through `MY_AUV_CONFIG`.

The two calibration files must be replaced with files produced by the camera  calibration process. The `.example` files show the expected whitespace-separated
format and contain illustrative values only; they are not valid calibration for
any particular camera.
