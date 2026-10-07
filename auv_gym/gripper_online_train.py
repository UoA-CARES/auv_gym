from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_MAIN_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_GRIPPER_ROOT = Path.home() / "gripper_things" / "Raining" / "gripper" / "gripper_gym" / "scripts"
_DEFAULT_GRIPPER_CONFIG_ROOT = Path.home() / "gripper_things" / "Raining" / "gripper" / "configs" / "gripper_configs"


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path.expanduser().resolve())
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def _ensure_dependency_paths(gripper_root: str | None = None) -> Path:
    _prepend_sys_path(_PACKAGE_ROOT)
    _prepend_sys_path(_MAIN_ROOT / "cares_reinforcement_learning")

    root = Path(gripper_root or os.environ.get("GRIPPER_RUNNER_ROOT", "") or _DEFAULT_GRIPPER_ROOT).expanduser()
    root = root.resolve()
    _prepend_sys_path(root)
    return root


_GRIPPER_ROOT = _ensure_dependency_paths()

from cares_lib.dynamixel.Gripper import GripperError  # noqa: E402
from cares_lib.dynamixel.gripper_configuration import GripperConfig  # noqa: E402
from cares_reinforcement_learning.memory.memory_factory import MemoryFactory  # noqa: E402
from cares_reinforcement_learning.util import configurations as cares_cfg  # noqa: E402
from cares_reinforcement_learning.util.network_factory import NetworkFactory  # noqa: E402
from cares_reinforcement_learning.util.training_context import ActionContext, TrainingContext  # noqa: E402
from configurations import GripperEnvironmentConfig  # noqa: E402
from environments.environment import EnvironmentError  # noqa: E402
from environments.environment_factory import EnvironmentFactory  # noqa: E402


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {p}")
    return payload


def _config_to_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "model_dump"):
        return dict(config.model_dump(exclude_none=True))
    if hasattr(config, "dict"):
        return dict(config.dict(exclude_none=True))
    return dict(config)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _checkpoint_kind(checkpoint: dict[str, Any], checkpoint_path: str) -> str:
    name = Path(checkpoint_path).name.upper()
    parent = Path(checkpoint_path).parent.name.upper()

    if "actor" in checkpoint and "target_critic" in checkpoint and "log_alpha" in checkpoint:
        return "SAC"
    if "actor" in checkpoint and "target_actor" in checkpoint and "critic" in checkpoint:
        critic = checkpoint.get("critic", {})
        if isinstance(critic, dict) and any(str(key).startswith("critic_net_") for key in critic):
            return "CTD4"
        if "CTD4" in name or "CTD4" in parent:
            return "CTD4"
        return "TD3"
    raise ValueError(
        "Unsupported checkpoint format. Expected a raw CARES TD3, CTD4, or SAC checkpoint "
        f"with actor/critic keys, got keys: {sorted(checkpoint.keys())}"
    )


def _load_checkpoint(path: str, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(str(Path(path).expanduser().resolve()), map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint dict, got {type(checkpoint)!r}")
    return checkpoint


def _infer_dims_from_checkpoint(checkpoint: dict[str, Any]) -> tuple[int, int]:
    actor = checkpoint.get("actor")
    if not isinstance(actor, dict):
        raise ValueError("Cannot infer dimensions: checkpoint is missing actor state dict.")

    first_weight = None
    last_weight = None
    for key, value in actor.items():
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            continue
        if first_weight is None:
            first_weight = value
        last_weight = value

    if first_weight is None or last_weight is None:
        raise ValueError("Cannot infer dimensions from actor state dict.")
    return int(first_weight.shape[1]), int(last_weight.shape[0])


def _load_alg_config(algorithm: str, config_path: str = ""):
    config_cls_name = f"{algorithm}Config"
    if not hasattr(cares_cfg, config_cls_name):
        raise ValueError(f"Missing CARES config class {config_cls_name}.")
    config_cls = getattr(cares_cfg, config_cls_name)

    if len(config_path.strip()) == 0:
        return config_cls(), "default"

    payload = _read_json(config_path)
    # train_vector_dreamer4 may write labels like TD3-t2750r2750-21. The CARES
    # factory still needs the base algorithm name to find create_TD3/create_SAC/etc.
    payload["algorithm"] = algorithm
    try:
        return config_cls(**payload), str(Path(config_path).expanduser().resolve())
    except Exception as exc:
        raise ValueError(f"Failed to parse CARES alg_config {config_path}: {exc}") from exc


def _move_agent_to_device(agent: Any, device: torch.device) -> None:
    if hasattr(agent, "device"):
        agent.device = device
    for _, value in vars(agent).items():
        if isinstance(value, nn.Module):
            value.to(device)
        elif isinstance(value, torch.Tensor):
            value.data = value.data.to(device)
            if value.grad is not None:
                value.grad = value.grad.to(device)


def _load_state_dict(module: Any, state_dict: dict[str, Any], label: str) -> None:
    try:
        module.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to load {label}; check that --alg_config matches the checkpoint architecture: {exc}") from exc


def _try_load_optimizer(optimizer: Any, state_dict: Any, label: str) -> bool:
    if optimizer is None or state_dict is None:
        return False
    try:
        optimizer.load_state_dict(state_dict)
        return True
    except Exception as exc:
        print(f"Warning: skipped loading {label} optimizer state: {exc}")
        return False


def _load_agent_weights(agent: Any, checkpoint: dict[str, Any], algorithm: str) -> dict[str, bool]:
    loaded = {
        "actor_optimizer": False,
        "critic_optimizer": False,
        "alpha_optimizer": False,
    }

    _load_state_dict(agent.actor_net, checkpoint["actor"], "actor")
    _load_state_dict(agent.critic_net, checkpoint["critic"], "critic")

    if hasattr(agent, "target_actor_net") and "target_actor" in checkpoint:
        _load_state_dict(agent.target_actor_net, checkpoint["target_actor"], "target_actor")
    if hasattr(agent, "target_critic_net") and "target_critic" in checkpoint:
        _load_state_dict(agent.target_critic_net, checkpoint["target_critic"], "target_critic")

    loaded["actor_optimizer"] = _try_load_optimizer(
        getattr(agent, "actor_net_optimiser", None),
        checkpoint.get("actor_optimizer"),
        "actor",
    )
    loaded["critic_optimizer"] = _try_load_optimizer(
        getattr(agent, "critic_net_optimiser", None),
        checkpoint.get("critic_optimizer"),
        "critic",
    )

    if algorithm == "SAC":
        if "log_alpha" in checkpoint and hasattr(agent, "log_alpha"):
            agent.log_alpha.data = torch.tensor(float(checkpoint["log_alpha"]), device=agent.device)
        loaded["alpha_optimizer"] = _try_load_optimizer(
            getattr(agent, "log_alpha_optimizer", None),
            checkpoint.get("log_alpha_optimizer"),
            "alpha",
        )

    if hasattr(agent, "learn_counter"):
        agent.learn_counter = int(checkpoint.get("learn_counter", 0))
    if hasattr(agent, "policy_noise"):
        agent.policy_noise = checkpoint.get("policy_noise", agent.policy_noise)
    if hasattr(agent, "action_noise"):
        agent.action_noise = checkpoint.get("action_noise", agent.action_noise)

    return loaded


def _create_agent(
    checkpoint: dict[str, Any],
    checkpoint_path: str,
    obs_dim: int,
    action_dim: int,
    alg_config_path: str,
    device: torch.device,
):
    algorithm = _checkpoint_kind(checkpoint, checkpoint_path)
    alg_config, alg_config_source = _load_alg_config(algorithm, alg_config_path)

    if bool(getattr(alg_config, "image_observation", 0)):
        raise ValueError(f"{algorithm} config expects image observations; this script supports vector state only.")

    agent = NetworkFactory().create_network(
        observation_size=int(obs_dim),
        action_num=int(action_dim),
        config=alg_config,
    )
    if agent is None:
        raise RuntimeError(f"Failed to create CARES {algorithm} agent.")
    _move_agent_to_device(agent, device)
    loaded = _load_agent_weights(agent, checkpoint, algorithm)
    return agent, alg_config, algorithm, alg_config_source, loaded


def _reset_env(environment: Any) -> np.ndarray:
    return np.asarray(environment.reset(), dtype=np.float32)


def _step_env(environment: Any, action_env: np.ndarray | list[float]) -> tuple[np.ndarray, float, bool, bool]:
    out = environment.step(action_env)
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, truncated = out
        return np.asarray(obs, dtype=np.float32), float(reward), bool(done), bool(truncated)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, done, truncated, _info = out
        return np.asarray(obs, dtype=np.float32), float(reward), bool(done), bool(truncated)
    raise ValueError("Unsupported environment.step(...) return signature.")


def _safe_reset(environment: Any) -> np.ndarray:
    try:
        return _reset_env(environment)
    except (EnvironmentError, GripperError) as exc:
        raise RuntimeError(f"Failed to reset gripper environment: {exc}") from exc


def _safe_step(environment: Any, action_env: np.ndarray | list[float]) -> tuple[np.ndarray, float, bool, bool]:
    try:
        return _step_env(environment, action_env)
    except (EnvironmentError, GripperError) as exc:
        raise RuntimeError(f"Failed to step gripper environment: {exc}") from exc


def _select_policy_action(agent: Any, state: np.ndarray, deterministic: bool) -> np.ndarray:
    context = ActionContext(
        state=np.asarray(state, dtype=np.float32),
        evaluation=bool(deterministic),
        available_actions=np.array([], dtype=np.float32),
    )
    action = agent.select_action_from_policy(context)
    return np.asarray(action, dtype=np.float32).reshape(-1)


def _train_updates(
    agent: Any,
    memory: Any,
    batch_size: int,
    updates: int,
    training_step: int,
    episode: int,
    episode_steps: int,
    episode_reward: float,
    episode_done: bool,
) -> dict[str, float]:
    metrics: dict[str, list[float]] = {}
    for _ in range(int(updates)):
        info = agent.train_policy(
            TrainingContext(
                memory=memory,
                batch_size=int(batch_size),
                training_step=int(training_step),
                episode=int(episode),
                episode_steps=int(episode_steps),
                episode_reward=float(episode_reward),
                episode_done=bool(episode_done),
            )
        )
        if not isinstance(info, dict):
            continue
        for key, value in info.items():
            if isinstance(value, (int, float, np.floating)):
                metrics.setdefault(key, []).append(float(value))
    return {key: float(np.mean(values)) for key, values in metrics.items() if values}


def _close_environment(environment: Any) -> None:
    try:
        if hasattr(environment, "close"):
            environment.close()
        elif hasattr(environment, "gripper") and hasattr(environment.gripper, "close"):
            environment.gripper.close()
    except Exception:
        pass


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    if not fields:
        fields = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_outputs(
    run_dir: Path,
    args: argparse.Namespace,
    checkpoint_kind: str,
    alg_config: Any,
    alg_config_source: str,
    optimizer_loaded: dict[str, bool],
    episode_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_csv(run_dir / "episode_metrics.csv", episode_rows)
    _save_csv(run_dir / "train_metrics.csv", train_rows)
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_kind": checkpoint_kind,
        "alg_config_source": alg_config_source,
        "optimizer_loaded": optimizer_loaded,
        "env_config": str(Path(args.env_config).expanduser().resolve()),
        "gripper_config": str(Path(args.gripper_config).expanduser().resolve()),
        "episodes": int(args.episodes),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "exploration_episodes": int(args.exploration_episodes),
        "batch_size": int(getattr(alg_config, "batch_size", args.batch_size)),
        "updates_per_step": int(args.updates_per_step),
        "algorithm_config": _config_to_dict(alg_config),
    }
    with (run_dir / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)


def _save_agent(agent: Any, path: Path, algorithm: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    agent.save_models(str(path), algorithm)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online real-world CARES fine-tuning for gripper checkpoints.",
    )
    parser.add_argument("--checkpoint", required=True, help="Raw CARES TD3/CTD4/SAC checkpoint .pth file.")
    parser.add_argument("--alg_config", default="", help="Optional matching CARES alg_config.json.")
    parser.add_argument("--env_config", default=str(_DEFAULT_GRIPPER_CONFIG_ROOT / "env_config.json"))
    parser.add_argument("--gripper_config", default=str(_DEFAULT_GRIPPER_CONFIG_ROOT / "gripper_config.json"))
    parser.add_argument("--gripper_runner_root", default=str(_DEFAULT_GRIPPER_ROOT))
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exploration_episodes", type=int, default=0)
    parser.add_argument("--deterministic_policy", action="store_true", help="Use deterministic/evaluation actions during training episodes.")
    parser.add_argument("--min_transitions_before_training", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=0, help="Override alg_config batch_size when > 0.")
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--results_dir", default="./real_online_train_gripper")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--save_every_episodes", type=int, default=5)
    parser.add_argument("--dry_run", action="store_true", help="Load checkpoint/config only; do not open gripper hardware.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _ensure_dependency_paths(args.gripper_runner_root)
    _set_seed(args.seed)

    device = _resolve_device(args.device)
    checkpoint_path = str(Path(args.checkpoint).expanduser().resolve())
    checkpoint = _load_checkpoint(checkpoint_path, device)
    inferred_obs_dim, inferred_action_dim = _infer_dims_from_checkpoint(checkpoint)

    if args.dry_run:
        agent, alg_config, algorithm, alg_config_source, loaded = _create_agent(
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            obs_dim=inferred_obs_dim,
            action_dim=inferred_action_dim,
            alg_config_path=args.alg_config,
            device=device,
        )
        print(
            f"Dry run OK: {Path(checkpoint_path).name} kind={algorithm} "
            f"obs={inferred_obs_dim} action={inferred_action_dim} "
            f"device={device} alg_config={alg_config_source} optimizer_loaded={loaded}"
        )
        del agent, alg_config
        return

    env_payload = _read_json(args.env_config)
    gripper_payload = _read_json(args.gripper_config)
    env_config = GripperEnvironmentConfig(**env_payload)
    gripper_config = GripperConfig(**gripper_payload)
    environment = EnvironmentFactory().create_environment(env_config, gripper_config)

    state0 = _safe_reset(environment)
    obs_dim = int(state0.reshape(-1).shape[0])
    action_dim = int(gripper_config.num_motors)
    if obs_dim != inferred_obs_dim or action_dim != inferred_action_dim:
        _close_environment(environment)
        raise ValueError(
            f"Environment/checkpoint dimension mismatch: env obs/action={obs_dim}/{action_dim}, "
            f"checkpoint obs/action={inferred_obs_dim}/{inferred_action_dim}."
        )

    agent, alg_config, algorithm, alg_config_source, loaded = _create_agent(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        obs_dim=obs_dim,
        action_dim=action_dim,
        alg_config_path=args.alg_config,
        device=device,
    )
    if args.batch_size > 0 and hasattr(alg_config, "batch_size"):
        alg_config.batch_size = int(args.batch_size)
    batch_size = int(getattr(alg_config, "batch_size", args.batch_size or 256))
    memory = MemoryFactory().create_memory(alg_config)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() or f"gripper_online_{algorithm}_{timestamp}"
    run_dir = Path(args.results_dir).expanduser().resolve() / run_name
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loaded {algorithm} checkpoint {Path(checkpoint_path).name} "
        f"(obs={obs_dim}, action={action_dim}, device={device}, alg_config={alg_config_source})"
    )

    episode_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    total_steps = 0

    try:
        for episode_idx in range(int(args.episodes)):
            state = state0 if episode_idx == 0 else _safe_reset(environment)
            episode_reward = 0.0
            steps_taken = 0
            is_exploration = episode_idx < int(args.exploration_episodes)

            for step_idx in range(int(args.max_steps)):
                if is_exploration:
                    action_env = environment.sample_action()
                    action = np.asarray(environment.normalize(action_env), dtype=np.float32).reshape(-1)
                else:
                    action = _select_policy_action(
                        agent=agent,
                        state=np.asarray(state, dtype=np.float32),
                        deterministic=bool(args.deterministic_policy),
                    )
                    action_env = environment.denormalize(action)

                next_state, reward, done, truncated = _safe_step(environment, action_env)
                terminal = bool(done or truncated)
                memory.add(state, action, reward, next_state, terminal)

                total_steps += 1
                steps_taken += 1
                episode_reward += float(reward)

                if (
                    not is_exploration
                    and len(memory) >= int(args.min_transitions_before_training)
                    and int(args.updates_per_step) > 0
                ):
                    metrics = _train_updates(
                        agent=agent,
                        memory=memory,
                        batch_size=batch_size,
                        updates=int(args.updates_per_step),
                        training_step=total_steps,
                        episode=episode_idx,
                        episode_steps=step_idx + 1,
                        episode_reward=episode_reward,
                        episode_done=terminal,
                    )
                    if metrics:
                        train_rows.append(
                            {
                                "episode_idx": episode_idx,
                                "step_idx": step_idx,
                                "total_steps": total_steps,
                                **metrics,
                            }
                        )

                state = next_state
                if terminal:
                    break

            episode_rows.append(
                {
                    "episode_idx": episode_idx,
                    "phase": "exploration" if is_exploration else "training",
                    "episode_reward": episode_reward,
                    "steps_taken": steps_taken,
                    "total_steps": total_steps,
                    "memory_size": len(memory),
                }
            )
            print(
                f"[episode {episode_idx} | {'explore' if is_exploration else 'train'}] "
                f"reward={episode_reward:.4f} steps={steps_taken} memory={len(memory)}"
            )

            if args.save_every_episodes > 0 and (episode_idx + 1) % int(args.save_every_episodes) == 0:
                _save_agent(agent, checkpoint_dir / f"episode_{episode_idx + 1}", algorithm)

        _save_agent(agent, checkpoint_dir / "final", algorithm)
    finally:
        _save_outputs(
            run_dir=run_dir,
            args=args,
            checkpoint_kind=algorithm,
            alg_config=alg_config,
            alg_config_source=alg_config_source,
            optimizer_loaded=loaded,
            episode_rows=episode_rows,
            train_rows=train_rows,
        )
        _close_environment(environment)

    print(f"Saved run outputs: {run_dir}")


if __name__ == "__main__":
    main()
