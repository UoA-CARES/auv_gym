import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import pydantic

from boxfish_lib.boxfish_configuration import BoxfishConfig
from auv_gym.tools.configurations import AUVEnvironmentConfig


def position_to_pixel(position, camera_matrix, reference_position=None):
    """
    Projects a 3D position to 2D pixel coordinates using the camera matrix.
    If reference_position is provided, it is added to position and its Z is used for scaling.
    If not, position is used directly.
    """
    if reference_position is not None:
        x = position[0] + reference_position[0]
        y = position[1] + reference_position[1]
        z = reference_position[2]
    else:
        x = position[0]
        y = position[1]
        z = position[2]
    pixel_x = camera_matrix[0, 0] * x / z + camera_matrix[0, 2]
    pixel_y = camera_matrix[1, 1] * y / z + camera_matrix[1, 2]
    return int(pixel_x), int(pixel_y)


def create_directories(local_results_path, folder_name):
    if not os.path.exists(local_results_path):
        os.makedirs(local_results_path)

    file_path = f"{local_results_path}/{folder_name}"

    if not os.path.exists(file_path):
        os.makedirs(file_path)
    if not os.path.exists(f"{file_path}/data"):
        os.makedirs(f"{file_path}/data")
    if not os.path.exists(
        "servo_errors"
    ):  # servo error still here because it's used by servo.py which shouldn't know the local storage
        os.makedirs("servo_errors")
    return file_path


def store_configs(file_path, parent_path, folder_name="configs"):
    if not os.path.isdir(f"{file_path + '/' + folder_name}"):
        os.mkdir(file_path + "/" + folder_name)

    for file_name in os.listdir(parent_path):
        # construct full file path
        source = parent_path + "/" + file_name

        destination = file_path + "/" + folder_name + "/" + file_name
        print(f"Destination: {destination}")

        # copy only files
        if os.path.isfile(source):
            shutil.copy(source, destination)
            print("copied", file_name)


def store_data(data, file_path, file_name):
    with open(f"{file_path}/data/{file_name}.txt", "a") as f:
        f.write(str(data) + "\n")


def plot_data(file_path, files):
    if type(files) is not list:
        files = [files]

    for file_name in files:
        datas = []
        with open(f"{file_path}/data/{file_name}.txt", "r") as file:
            for line in file:
                data = float(line.strip())
                datas.append(data)

        plt.plot(datas)
        plt.xlabel("Episode")
        plt.ylabel(f"{file_name}")
        plt.title(f"{file_name}")
        plt.savefig(f"{file_path}/{file_name}")
        plt.close()


def plot_data_time(file_path, files, file_name_average_reward, file_name_time):
    average_reward = []
    time = []
    if type(files) is not list:
        files = [files]

    with open(f"{file_path}/data/{file_name_average_reward}.txt", "r") as file:
        for line in file:
            data = float(line.strip())
            average_reward.append(data)

    with open(f"{file_path}/data/{file_name_time}.txt", "r") as file:
        for line in file:
            data = float(line.strip())
            time.append(data)

        plt.plot(time, average_reward)
        plt.xlabel("Time")
        plt.ylabel(f"{file_name_average_reward}")
        plt.title("Average Reward vs Time")
        plt.savefig(f"{file_path}/reward_average_vs_time")
        plt.close()


def slack_post_plot(environment, slack_bot, file_path, plots):
    if type(plots) is not list:
        plots = [plots]

    for plot_name in plots:
        if os.path.exists(f"{file_path}/{plot_name}.png"):
            slack_bot.upload_file(
                "#cares-chat-bot",
                f"#{environment.gripper.gripper_id}: {plot_name}",
                f"{file_path}/",
                f"{plot_name}.png",
            )
        else:
            slack_bot.post_message(
                "#cares-chat-bot",
                f"#{environment.gripper.gripper_id}: {plot_name} plot not ready yet or doesn't exist",
            )


def save_evaluation_values(data_eval_reward, filename, file_path):
    data = pd.DataFrame.from_dict(data_eval_reward)
    data.to_csv(f"{file_path}/data/{filename}_evaluation", index=False)
    data.plot(x="step", y="avg_episode_reward", title="Evaluation Reward Curve")
    plt.savefig(f"{file_path}/data/{filename}_evaluation.png")
    plt.close()

def save_reward(source, destination):
    shutil.copy2(source, destination)

    print(f'File copied from {source} to {destination}')


def load_auv_env_config(config_path: str) -> AUVEnvironmentConfig:
    try:
        return pydantic.parse_file_as(path=config_path, type_=AUVEnvironmentConfig)
    except FileNotFoundError as e:
        error_msg = f"AUV environment config file not found: {config_path}"
        raise FileNotFoundError(error_msg) from e
    except Exception as e:
        error_msg = f"Failed to load AUV environment config from {config_path}: {e}"
        raise ValueError(error_msg) from e
    

def load_auv_config(config_path: str) -> BoxfishConfig:
    try:
        return pydantic.parse_file_as(path=config_path, type_=BoxfishConfig)
    except FileNotFoundError as e:
        error_msg = f"AUV config file not found: {config_path}"
        raise FileNotFoundError(error_msg) from e
    except Exception as e:
        error_msg = f"Failed to load AUV config from {config_path}: {e}"
        raise ValueError(error_msg) from e