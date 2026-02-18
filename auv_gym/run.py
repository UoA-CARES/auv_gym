import logging
import os
import yaml

logging.basicConfig(level=logging.INFO)

# import traceback

# orig_info = logging.info

# def custom_info(msg, *args, **kwargs):
#     print(">>> logging.info called. Stack trace:")
#     traceback.print_stack()
#     orig_info(msg, *args, **kwargs)

# logging.info = custom_info

from tools.configurations import AUVEnvironmentConfig
from auv_trainer import AUVTrainer
from tools.utils import save_reward

from boxfish_lib.boxfish_configuration import BoxfishConfig

from cares_reinforcement_learning.util import Record
from cares_reinforcement_learning.util import RLParser
from cares_reinforcement_learning.util.rl_parser import RunConfig
from cares_reinforcement_learning.util import configurations as cares_cfg
from cares_reinforcement_learning.util import helpers as hlp



def main():
    parser = RLParser(AUVEnvironmentConfig)
    parser.add_configuration("auv_config", BoxfishConfig)

    configurations = parser.parse_args()
    run_config: RunConfig = configurations["run_config"]
    env_config: AUVEnvironmentConfig = configurations["env_config"]
    training_config: cares_cfg.TrainingConfig = configurations["train_config"]
    alg_config: cares_cfg.AlgorithmConfig = configurations["alg_config"]
    auv_config: BoxfishConfig = configurations["auv_config"]

    if env_config.is_debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.info(f"------------------------------------------")
    logging.info(
        f"\n**************\n"
        f"Environment Config:\n"
        f"**************\n"
        f"{yaml.dump(env_config.dict(), default_flow_style=False)}\n"
        f"\n**************\n"
        f"Algorithm Config:\n"
        f"**************\n"
        f"{yaml.dump(alg_config.dict(), default_flow_style=False)}"
        f"\n**************\n"
        f"Training Config:\n"
        f"**************\n"
        f"{yaml.dump(training_config.dict(), default_flow_style=False)}"
        f"\n**************\n"
        f"AUV Config:\n"
        f"**************\n"
        f"{yaml.dump(auv_config.dict(), default_flow_style=False)}"
    )
    logging.info(f"------------------------------------------")

    run_name = input(
        "Double check your experiment configurations :) Press ENTER to continue. (Optional - Enter a name for this run)\n"
    )

    logging.info(f"Command: {run_config.command}")

    format_str="{algorithm}/{algorithm}-{seed}-{run_name}-{date}-{domain_task}"

    base_log_dir = Record.create_base_directory(
        domain=env_config.domain,
        task=env_config.task,
        gym="auv_gym",
        algorithm=alg_config.algorithm,
        run_name=run_name,
        seed=training_config.seeds[0],
        base_dir="../boxfish_logs",
        format_str=format_str,
    )

    record = Record(
        base_directory=base_log_dir,
        task=env_config.task,
        algorithm=alg_config.algorithm,
        agent=None,
    )

    record.save_configurations(configurations)
    save_reward(source=f"{os.path.dirname(os.path.abspath(__file__))}/environments/reward_config.py", destination=base_log_dir)

    for iteration, seed in enumerate(training_config.seeds):
        logging.info(f"Iteration {iteration + 1}/{len(training_config.seeds)} with seed {seed}")

        logging.info("Setting up Seeds")
        hlp.set_seed(seed)

        auv_trainer = AUVTrainer(
            env_config=env_config,
            training_config=training_config,
            alg_config=alg_config,
            auv_config=auv_config,
            record=record,
        )

        record.set_sub_directory(seed)
        record.set_agent(auv_trainer.agent)

        if run_config.command == "train":
            # if env_config.reward_function == "transfer":
            #     # auv_trainer.agent.load_models("/home/yxin683/Desktop/transfer_model",f"{alg_config.algorithm}_{training_config.seeds[0]}")
            #     gripper_teacher.agent.load_models("/home/yxin683/Desktop/transfer_model",f"{alg_config.algorithm}_{training_config.seeds[0]}")
            auv_trainer.train()
        elif run_config.command == "evaluate":

            # load trained model
            auv_trainer.agent.load_models('./checkpoints', 'SAC_highest')
            
            # load dreamer model
            from tools.dreamer4_reward_fix import ImprovedDreamer4Agent
            agent = ImprovedDreamer4Agent(obs_dim=6, action_dim=1)
            agent.load('./checkpoints/dreamer4_fixed.pt')
            auv_trainer.agent = agent
    
            auv_trainer.evaluation_loop(200)

        if len(training_config.seeds) > 0:
            logging.warning("Multiple seeds are not yet supported.")
            break

if __name__ == "__main__":
    main()
