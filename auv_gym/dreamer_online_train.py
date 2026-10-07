from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_MAIN_ROOT = Path(__file__).resolve().parents[2]


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path.resolve())
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def _iter_gptdream_candidates() -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path | None) -> None:
        if path is None:
            return
        resolved = str(path.expanduser().resolve())
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(Path(resolved))

    env_candidates = [
        os.environ.get("GPTDREAM_ROOT"),
        os.environ.get("DREAMER4_ROOT"),
    ]
    for raw_path in env_candidates:
        if raw_path:
            add(Path(raw_path))

    home = Path.home()
    add(home / "gptdream")
    add(home / "Documents" / "gptdream")

    roots_to_scan = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    for root in roots_to_scan:
        for parent in [root, *root.parents]:
            add(parent / "gptdream")
            add(parent / "main" / "gptdream")

    return candidates


def _ensure_dependency_paths() -> None:
    _prepend_sys_path(_PACKAGE_ROOT)
    _prepend_sys_path(_MAIN_ROOT / "boxfish_lib" / "boxfish_lib")
    _prepend_sys_path(_MAIN_ROOT / "cares_reinforcement_learning")

    if importlib.util.find_spec("dreamer4_vector") is not None:
        pass
    else:
        for candidate in _iter_gptdream_candidates():
            if (candidate / "dreamer4_vector").is_dir():
                _prepend_sys_path(candidate)
                if importlib.util.find_spec("dreamer4_vector") is not None:
                    break

    if importlib.util.find_spec("dreamer4_vector") is None:
        raise ModuleNotFoundError(
            "Could not find the 'dreamer4_vector' package. "
            "Set GPTDREAM_ROOT to the folder that contains dreamer4_vector, "
            "for example: export GPTDREAM_ROOT=/path/to/gptdream"
        )


_ensure_dependency_paths()

import numpy as np
import torch
import torch.nn.functional as F

from auv_gym.environments.environment_factory import EnvironmentFactory
from dreamer4_vector.checkpoint import load_checkpoint, save_checkpoint
from dreamer4_vector.config import TrainConfig
from dreamer4_vector.models import PolicyHead, RewardHead, ValueHead, WorldModelDynamics
from dreamer4_vector.replay import Episode, ReplayBuffer, load_cares_memory_buffer, load_cares_memory_buffers
from dreamer4_vector.trainer import (
    _normalize_action,
    _normalize_reward,
    _normalize_state,
    _policy_losses,
    _sample_next_state_shortcut,
    world_model_shortcut_loss,
)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _reset_env(environment: Any, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    reset_info: dict[str, Any] = {}
    if seed is not None:
        _set_seed(seed)
        if hasattr(environment, "set_seed"):
            try:
                environment.set_seed(seed)
            except Exception:
                pass

    try:
        out = environment.reset(seed=seed)
    except TypeError:
        try:
            out = environment.reset(training=False)
        except TypeError:
            out = environment.reset()

    if isinstance(out, tuple):
        obs = out[0]
        if len(out) > 1 and isinstance(out[1], dict):
            reset_info = out[1]
    else:
        obs = out
    return np.asarray(obs, dtype=np.float32), reset_info


def _step_env(environment: Any, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
    out = environment.step(action)
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return np.asarray(obs, dtype=np.float32), float(reward), bool(done), info
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated or truncated), info
    raise ValueError("Unsupported environment.step(...) return signature.")


def _clip_normalized_action(action: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)


def _extract_goal_signature(environment: Any, reset_info: dict[str, Any]) -> str:
    goal_keys = [
        "goal",
        "goal_id",
        "chosen_goal",
        "chosen_marker_id",
        "target",
        "target_id",
        "target_yaw",
    ]
    payload: dict[str, Any] = {}
    for key in goal_keys:
        if key in reset_info:
            payload[key] = reset_info[key]

    env_obj = environment
    for _ in range(3):
        if hasattr(env_obj, "chosen_marker_id"):
            payload["chosen_marker_id_attr"] = getattr(env_obj, "chosen_marker_id")
            break
        if hasattr(env_obj, "env"):
            env_obj = getattr(env_obj, "env")
            continue
        break

    if len(payload) == 0:
        return ""
    try:
        return json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        return str(payload)


def _load_train_config(checkpoint: dict[str, Any], args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    train_payload = checkpoint.get("train_config", {})
    if isinstance(train_payload, dict):
        for key, value in train_payload.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    overrides = {
        "device": args.device,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "lr_world": args.lr_world,
        "lr_agent": args.lr_agent,
        "lr_rl_policy": args.lr_rl_policy,
        "lr_rl_value": args.lr_rl_value,
        "imagination_horizon": args.imagination_horizon,
        "gamma": args.gamma,
        "lambda_": args.lambda_,
        "pmpo_alpha": args.pmpo_alpha,
        "pmpo_beta": args.pmpo_beta,
        "bc_loss_scale": args.bc_loss_scale,
        "reward_loss_scale": args.reward_loss_scale,
        "grad_clip": args.grad_clip,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)

    cfg.rl_algo = "pmpo"
    return cfg


def _load_bootstrap_episodes(args: argparse.Namespace) -> list[Episode]:
    if args.bootstrap_buffer_dir:
        data = load_cares_memory_buffers(
            pkl_paths=[
                str(path)
                for path in sorted(Path(args.bootstrap_buffer_dir).expanduser().glob(args.bootstrap_buffer_glob))
            ],
            fixed_episode_len=args.bootstrap_fixed_episode_len or None,
        )
        return list(data["buffer"].episodes)

    if args.bootstrap_buffer_pkl:
        data = load_cares_memory_buffer(
            pkl_path=args.bootstrap_buffer_pkl,
            fixed_episode_len=args.bootstrap_fixed_episode_len or None,
        )
        return list(data["buffer"].episodes)

    return []


def _describe_checkpoint_format(path: str, device: str) -> tuple[str, list[str]]:
    map_location = torch.device(device)
    raw_ckpt = torch.load(path, map_location=map_location)
    if not isinstance(raw_ckpt, dict):
        return type(raw_ckpt).__name__, []

    keys = sorted(str(key) for key in raw_ckpt.keys())

    if "meta" in raw_ckpt and "policy_state_dict" in raw_ckpt:
        return "dreamer", keys
    if raw_ckpt.get("format") == "dreamer4_vector_phase12_cache_v1" and "phase12_state" in raw_ckpt:
        return "dreamer_phase12_cache", keys
    if (
        "actor" in raw_ckpt
        and "target_actor" in raw_ckpt
        and "critic" in raw_ckpt
        and isinstance(raw_ckpt["critic"], dict)
        and any(str(k).startswith("critic_net_") for k in raw_ckpt["critic"].keys())
    ):
        return "cares_ctd4", keys
    if "actor" in raw_ckpt and "target_critic" in raw_ckpt and "log_alpha" in raw_ckpt:
        return "cares_sac", keys
    if "actor" in raw_ckpt and "target_actor" in raw_ckpt:
        return "cares_td3", keys
    if "actor" in raw_ckpt and "critic" in raw_ckpt:
        return "cares_actor_critic", keys
    return "unknown_dict", keys


class OnlineDreamerTrainer:
    def __init__(self, checkpoint_path: str, args: argparse.Namespace):
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.device = _resolve_device(args.device)
        try:
            self.checkpoint = load_checkpoint(self.checkpoint_path, device=str(self.device))
        except ValueError as exc:
            checkpoint_kind, checkpoint_keys = _describe_checkpoint_format(
                self.checkpoint_path,
                device=str(self.device),
            )
            if checkpoint_kind.startswith("cares_"):
                key_text = ", ".join(checkpoint_keys)
                raise ValueError(
                    "dreamer_online_train.py only supports Dreamer-vector checkpoints that contain "
                    "world/policy/reward/value state dicts. "
                    f"The file '{self.checkpoint_path}' looks like a CARES {checkpoint_kind.removeprefix('cares_').upper()} "
                    f"checkpoint instead (keys: [{key_text}]). "
                    "Use the standard CARES training pipeline to continue training that policy in the real world, "
                    "for example via main/gymnasium_envrionments/scripts/run.py with the matching algorithm and "
                    "--model_path pointing to the checkpoint directory."
                ) from exc
            raise
        self.cfg = _load_train_config(self.checkpoint, args)

        meta = self.checkpoint["meta"]
        self.state_dim = int(meta["state_dim"])
        self.action_dim = int(meta["action_dim"])
        self.discrete_action = bool(meta["discrete_action"])
        if self.discrete_action:
            raise ValueError("Online trainer currently supports continuous-action checkpoints only.")

        self.world = WorldModelDynamics(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_action=self.discrete_action,
            d_model=self.cfg.d_model,
            n_heads=self.cfg.n_heads,
            n_layers=self.cfg.n_layers,
            ff_mult=self.cfg.ff_mult,
            dropout=self.cfg.dropout,
        ).to(self.device)
        self.policy = PolicyHead(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            discrete_action=self.discrete_action,
        ).to(self.device)
        self.reward = RewardHead(self.state_dim).to(self.device)
        self.value = ValueHead(self.state_dim).to(self.device)

        self.world.load_state_dict(self.checkpoint["world_state_dict"], strict=True)
        self.policy.load_state_dict(self.checkpoint["policy_state_dict"], strict=True)
        self.reward.load_state_dict(self.checkpoint["reward_state_dict"], strict=True)
        self.value.load_state_dict(self.checkpoint["value_state_dict"], strict=True)

        self.opt_world = torch.optim.Adam(self.world.parameters(), lr=self.cfg.lr_world)
        self.opt_agent = torch.optim.Adam(
            list(self.world.parameters()) + list(self.policy.parameters()) + list(self.reward.parameters()),
            lr=self.cfg.lr_agent,
        )
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr_rl_policy)
        self.opt_value = torch.optim.Adam(self.value.parameters(), lr=self.cfg.lr_rl_value)

        norm = self.checkpoint["normalization"]
        self.state_mean_eval = torch.as_tensor(norm["state_mean"], dtype=torch.float32, device=self.device).view(
            1, self.state_dim
        )
        self.state_std_eval = torch.as_tensor(norm["state_std"], dtype=torch.float32, device=self.device).view(
            1, self.state_dim
        )
        self.state_mean_train = self.state_mean_eval.view(1, 1, self.state_dim)
        self.state_std_train = self.state_std_eval.view(1, 1, self.state_dim)
        self.state_clip = float(norm.get("state_clip", self.cfg.state_clip))

        action_mean = norm.get("action_mean")
        action_std = norm.get("action_std")
        self.action_mean_eval = (
            None
            if action_mean is None
            else torch.as_tensor(action_mean, dtype=torch.float32, device=self.device).view(1, self.action_dim)
        )
        self.action_std_eval = (
            None
            if action_std is None
            else torch.as_tensor(action_std, dtype=torch.float32, device=self.device).view(1, self.action_dim)
        )
        self.action_mean_train = None if self.action_mean_eval is None else self.action_mean_eval.view(1, 1, -1)
        self.action_std_train = None if self.action_std_eval is None else self.action_std_eval.view(1, 1, -1)

        self.reward_mean_train = torch.as_tensor(
            norm.get("reward_mean", 0.0), dtype=torch.float32, device=self.device
        )
        self.reward_std_train = torch.as_tensor(
            norm.get("reward_std", 1.0), dtype=torch.float32, device=self.device
        ).clamp_min(1e-3)

        self.bootstrap_episodes = _load_bootstrap_episodes(args)
        self.online_episodes: list[Episode] = []

    @property
    def total_episodes(self) -> int:
        return len(self.bootstrap_episodes) + len(self.online_episodes)

    @property
    def total_transitions(self) -> int:
        return int(sum(len(ep.actions) for ep in self.bootstrap_episodes + self.online_episodes))

    @property
    def max_episode_len(self) -> int:
        if self.total_episodes == 0:
            return 0
        return self._make_buffer().max_episode_len

    def _make_buffer(self) -> ReplayBuffer:
        return ReplayBuffer(self.bootstrap_episodes + self.online_episodes, discrete_action=False)

    @torch.no_grad()
    def get_action(self, obs: np.ndarray, deterministic: bool) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, self.state_dim)
        state_n = torch.clamp((obs_t - self.state_mean_eval) / self.state_std_eval, -self.state_clip, self.state_clip)
        dist = self.policy.dist(state_n)
        action_n = dist.mean if deterministic else dist.sample()
        if self.action_mean_eval is not None and self.action_std_eval is not None:
            action = action_n * self.action_std_eval + self.action_mean_eval
        else:
            action = action_n
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def add_episode(self, states: list[np.ndarray], actions: list[np.ndarray], rewards: list[float]) -> None:
        if len(actions) == 0:
            return
        episode = Episode(
            states=np.asarray(states, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            continues=np.ones(len(actions), dtype=np.float32),
        )
        if len(episode.continues) > 0:
            episode.continues[-1] = 0.0
        self.online_episodes.append(episode)

    def can_train(self, min_transitions: int) -> bool:
        if self.total_transitions < min_transitions:
            return False
        if self.total_episodes == 0:
            return False
        return self.max_episode_len >= 1

    def _supervised_update(self, buffer: ReplayBuffer) -> dict[str, float]:
        seq_len = min(int(self.cfg.seq_len), buffer.max_episode_len)
        batch = buffer.sample_batch(int(self.cfg.batch_size), seq_len, self.device)
        states_n = _normalize_state(batch["states"], self.state_mean_train, self.state_std_train, self.cfg.state_clip)
        actions_n = _normalize_action(
            batch["actions"],
            self.action_mean_train,
            self.action_std_train,
            self.cfg.action_clip,
            buffer.discrete_action,
        )
        rewards_n = _normalize_reward(batch["rewards"], self.reward_mean_train, self.reward_std_train, self.cfg.reward_clip)

        wm_out = world_model_shortcut_loss(
            self.world,
            states_n,
            actions_n,
            self.cfg.k_max,
            min_denom=self.cfg.min_denom,
            velocity_clip=self.cfg.velocity_clip,
            loss_clip=self.cfg.wm_loss_clip,
        )

        state_now = states_n[:, :-1].reshape(-1, self.state_dim)
        state_next = states_n[:, 1:].reshape(-1, self.state_dim)
        action = actions_n.reshape(-1, self.action_dim)
        reward_target = rewards_n.reshape(-1)

        bc_loss = -self.policy.log_prob(state_now, action).mean()
        reward_pred = self.reward(state_next)
        reward_loss = F.smooth_l1_loss(reward_pred, reward_target)
        total_loss = wm_out["loss"] + self.cfg.bc_loss_scale * bc_loss + self.cfg.reward_loss_scale * reward_loss

        self.opt_agent.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.world.parameters()) + list(self.policy.parameters()) + list(self.reward.parameters()),
            self.cfg.grad_clip,
        )
        self.opt_agent.step()

        return {
            "supervised_total": float(total_loss.item()),
            "world_model": float(wm_out["loss"].item()),
            "behavior_cloning": float(bc_loss.item()),
            "reward_loss": float(reward_loss.item()),
        }

    def _imagination_update(self, buffer: ReplayBuffer) -> dict[str, float]:
        prior = deepcopy(self.policy).to(self.device).eval()
        for param in prior.parameters():
            param.requires_grad_(False)

        start = buffer.sample_start_states(int(self.cfg.batch_size), self.device)
        start = _normalize_state(start.unsqueeze(1), self.state_mean_train, self.state_std_train, self.cfg.state_clip)
        start = start.squeeze(1)

        states = [start]
        actions = []
        rewards = []
        values = []

        for _ in range(int(self.cfg.imagination_horizon)):
            state = states[-1]
            values.append(self.value(state))
            action, _ = self.policy.sample(state)
            next_state = _sample_next_state_shortcut(self.world, state, action, self.cfg.k_infer)
            next_state = torch.clamp(next_state, -self.cfg.state_clip, self.cfg.state_clip)
            reward = self.reward(next_state)
            actions.append(action)
            rewards.append(reward)
            states.append(next_state)
        values.append(self.value(states[-1]))

        states_t = torch.stack(states, dim=1)
        actions_t = torch.stack(actions, dim=1)
        rewards_t = torch.stack(rewards, dim=1)
        values_t = torch.stack(values, dim=1)

        returns = torch.zeros_like(values_t)
        returns[:, -1] = values_t[:, -1].detach()
        for step in reversed(range(int(self.cfg.imagination_horizon))):
            bootstrap = (1.0 - self.cfg.lambda_) * values_t[:, step + 1].detach() + self.cfg.lambda_ * returns[:, step + 1]
            returns[:, step] = rewards_t[:, step].detach() + self.cfg.gamma * bootstrap

        value_pred = values_t[:, :-1]
        value_target = returns[:, :-1].detach()
        value_loss = F.smooth_l1_loss(value_pred, value_target)

        flat_states = states_t[:, :-1].reshape(-1, self.state_dim)
        flat_actions = actions_t.reshape(-1, self.action_dim)
        advantages = (returns[:, :-1] - values_t[:, :-1].detach()).reshape(-1)
        advantages = torch.clamp(advantages, -self.cfg.adv_clip, self.cfg.adv_clip)
        policy_out = _policy_losses(
            policy=self.policy,
            prior=prior,
            states=flat_states,
            actions=flat_actions,
            advantages=advantages,
            alpha=self.cfg.pmpo_alpha,
            beta=self.cfg.pmpo_beta,
        )

        self.opt_value.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.cfg.grad_clip)
        self.opt_value.step()

        self.opt_policy.zero_grad(set_to_none=True)
        policy_out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
        self.opt_policy.step()

        return {
            "policy_loss": float(policy_out["loss"].item()),
            "value_loss": float(value_loss.item()),
            "policy_pos": float(policy_out["pos_term"].item()),
            "policy_neg": float(policy_out["neg_term"].item()),
            "policy_kl": float(policy_out["kl_term"].item()),
            "imagined_reward_mean": float(rewards_t.mean().item()),
        }

    def train_burst(
        self,
        supervised_updates: int,
        imagination_updates: int,
    ) -> dict[str, float]:
        buffer = self._make_buffer()
        metrics: dict[str, list[float]] = {}

        for _ in range(int(supervised_updates)):
            update = self._supervised_update(buffer)
            for key, value in update.items():
                metrics.setdefault(key, []).append(value)

        for _ in range(int(imagination_updates)):
            update = self._imagination_update(buffer)
            for key, value in update.items():
                metrics.setdefault(key, []).append(value)

        return {key: float(np.mean(values)) for key, values in metrics.items()}

    def build_checkpoint_result(self) -> dict[str, Any]:
        return {
            "world": self.world,
            "policy": self.policy,
            "reward": self.reward,
            "value": self.value,
            "config": self.cfg,
            "device": str(self.device),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "discrete_action": self.discrete_action,
            "normalization": {
                "state_mean": self.state_mean_eval.squeeze(0).detach().cpu(),
                "state_std": self.state_std_eval.squeeze(0).detach().cpu(),
                "action_mean": None if self.action_mean_eval is None else self.action_mean_eval.squeeze(0).detach().cpu(),
                "action_std": None if self.action_std_eval is None else self.action_std_eval.squeeze(0).detach().cpu(),
                "reward_mean": self.reward_mean_train.detach().cpu(),
                "reward_std": self.reward_std_train.detach().cpu(),
                "state_clip": self.cfg.state_clip,
                "action_clip": self.cfg.action_clip,
                "reward_clip": self.cfg.reward_clip,
            },
        }


def _select_exploration_action(
    trainer: OnlineDreamerTrainer,
    environment: Any,
    obs: np.ndarray,
    strategy: str,
    policy_noise_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    if strategy == "env_random":
        if not hasattr(environment, "sample_action"):
            raise ValueError(
                "Exploration strategy 'env_random' requires environment.sample_action()."
            )
        action_env = np.asarray(environment.sample_action(), dtype=np.float32).reshape(-1)
        action = np.asarray(environment.normalize(action_env), dtype=np.float32).reshape(-1)
        return action, action_env

    if strategy == "policy_stochastic":
        action = trainer.get_action(obs, deterministic=False)
        action_env = np.asarray(environment.denormalize(action), dtype=np.float32).reshape(-1)
        return action, action_env

    if strategy == "policy_noise":
        action = trainer.get_action(obs, deterministic=True)
        noise = np.random.normal(
            loc=0.0,
            scale=float(policy_noise_std),
            size=action.shape,
        ).astype(np.float32)
        action = _clip_normalized_action(action + noise)
        action_env = np.asarray(environment.denormalize(action), dtype=np.float32).reshape(-1)
        return action, action_env

    raise ValueError(f"Unknown exploration strategy: {strategy}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online real-world fine-tuning for saved Dreamer-vector checkpoints.",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved Dreamer checkpoint (.pt/.pth).")
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | cuda:0 ...")
    parser.add_argument("--env_family", type=str, default="boxfish", help="Environment family.")
    parser.add_argument("--env_task", type=str, default="FiveMarkerSpin", help="Environment task name.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of real-world episodes to run.")
    parser.add_argument("--max_steps", type=int, default=20, help="Maximum steps per episode.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed used for resets.")
    parser.add_argument("--episode_seed_stride", type=int, default=1, help="Seed increment per episode.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using the deterministic mean.")
    parser.add_argument(
        "--exploration_episodes",
        type=int,
        default=0,
        help="Number of warmup episodes to collect before any training updates start.",
    )
    parser.add_argument(
        "--exploration_strategy",
        type=str,
        default="env_random",
        choices=["env_random", "policy_stochastic", "policy_noise"],
        help=(
            "How to act during warmup exploration episodes: "
            "env_random uses environment.sample_action(), "
            "policy_stochastic samples from the policy, "
            "policy_noise adds small noise to the deterministic policy action."
        ),
    )
    parser.add_argument(
        "--exploration_policy_noise_std",
        type=float,
        default=0.10,
        help="Standard deviation of normalized-action Gaussian noise for --exploration_strategy policy_noise.",
    )
    parser.add_argument(
        "--no_env_denormalize",
        action="store_true",
        help="Do not call environment.denormalize(action) before step.",
    )
    parser.add_argument(
        "--min_transitions_before_training",
        type=int,
        default=128,
        help="Do not start gradient updates until this many transitions are available.",
    )
    parser.add_argument(
        "--supervised_updates_per_episode",
        type=int,
        default=10,
        help="Number of real-data world/reward/policy updates after each episode.",
    )
    parser.add_argument(
        "--imagination_updates_per_episode",
        type=int,
        default=10,
        help="Number of imagination RL updates after each episode.",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Override checkpoint batch size.")
    parser.add_argument("--seq_len", type=int, default=None, help="Override checkpoint sequence length.")
    parser.add_argument("--imagination_horizon", type=int, default=None, help="Override checkpoint imagination horizon.")
    parser.add_argument("--gamma", type=float, default=None, help="Override checkpoint gamma.")
    parser.add_argument("--lambda_", type=float, default=None, help="Override checkpoint lambda.")
    parser.add_argument("--pmpo_alpha", type=float, default=None, help="Override checkpoint PMPO alpha.")
    parser.add_argument("--pmpo_beta", type=float, default=None, help="Override checkpoint PMPO beta.")
    parser.add_argument("--bc_loss_scale", type=float, default=None, help="Override behavior-cloning loss scale.")
    parser.add_argument("--reward_loss_scale", type=float, default=None, help="Override reward loss scale.")
    parser.add_argument("--grad_clip", type=float, default=None, help="Override gradient clip value.")
    parser.add_argument("--lr_world", type=float, default=None, help="Override world-model learning rate.")
    parser.add_argument("--lr_agent", type=float, default=None, help="Override joint finetuning learning rate.")
    parser.add_argument("--lr_rl_policy", type=float, default=None, help="Override policy RL learning rate.")
    parser.add_argument("--lr_rl_value", type=float, default=None, help="Override value learning rate.")
    parser.add_argument(
        "--bootstrap_buffer_pkl",
        type=str,
        default="",
        help="Optional CARES memory buffer .pkl to seed the online replay with.",
    )
    parser.add_argument(
        "--bootstrap_buffer_dir",
        type=str,
        default="",
        help="Optional directory of CARES memory buffer .pkl files to seed the online replay with.",
    )
    parser.add_argument(
        "--bootstrap_buffer_glob",
        type=str,
        default="*.pkl",
        help="Glob used with --bootstrap_buffer_dir.",
    )
    parser.add_argument(
        "--bootstrap_fixed_episode_len",
        type=int,
        default=0,
        help="Optional fixed episode length when splitting bootstrap buffers.",
    )
    parser.add_argument("--results_dir", type=str, default="./real_online_train", help="Directory for logs and saved checkpoints.")
    parser.add_argument("--run_name", type=str, default="", help="Optional run name for the output subdirectory.")
    parser.add_argument(
        "--save_every_episodes",
        type=int,
        default=5,
        help="Checkpoint save interval in episodes. Set 0 to disable intermediate saves.",
    )
    parser.add_argument(
        "--success_step_reward",
        type=float,
        default=4.0,
        help="Reward threshold for counting steps at goal.",
    )
    parser.add_argument(
        "--success_episode_reward",
        type=float,
        default=4.0,
        help="Episode reward threshold for counting success.",
    )
    return parser.parse_args()


def _save_run_outputs(
    run_dir: Path,
    args: argparse.Namespace,
    trainer: OnlineDreamerTrainer,
    episode_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        fields = sorted({key for row in episode_rows for key in row.keys()})
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in episode_rows:
            writer.writerow(row)

    with (run_dir / "train_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        fields = sorted({key for row in train_rows for key in row.keys()})
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in train_rows:
            writer.writerow(row)

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "checkpoint": trainer.checkpoint_path,
        "episodes": int(args.episodes),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "stochastic": bool(args.stochastic),
        "env_family": args.env_family,
        "env_task": args.env_task,
        "exploration_episodes": int(args.exploration_episodes),
        "exploration_strategy": args.exploration_strategy,
        "exploration_policy_noise_std": float(args.exploration_policy_noise_std),
        "bootstrap_episodes": len(trainer.bootstrap_episodes),
        "online_episodes": len(trainer.online_episodes),
        "total_transitions": trainer.total_transitions,
        "train_config": asdict(trainer.cfg),
    }
    with (run_dir / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)

    trainer = OnlineDreamerTrainer(args.checkpoint, args)
    env_factory = EnvironmentFactory()
    environment = env_factory.create_environment(args.env_family, args.env_task)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() or f"online_{timestamp}"
    run_dir = Path(args.results_dir).expanduser().resolve() / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loaded checkpoint {Path(trainer.checkpoint_path).name} "
        f"(obs={trainer.state_dim}, action={trainer.action_dim}, device={trainer.device})"
    )
    print(
        f"Bootstrap episodes: {len(trainer.bootstrap_episodes)} "
        f"total bootstrap transitions: {trainer.total_transitions}"
    )

    episode_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []

    try:
        for episode_idx in range(int(args.episodes)):
            episode_seed = int(args.seed + episode_idx * args.episode_seed_stride)
            obs, reset_info = _reset_env(environment, seed=episode_seed)
            goal_signature = _extract_goal_signature(environment, reset_info)
            is_exploration_episode = episode_idx < int(args.exploration_episodes)

            states = [obs.copy()]
            actions: list[np.ndarray] = []
            rewards: list[float] = []
            episode_reward = 0.0
            steps_at_goal = 0
            steps_taken = 0

            for _ in range(int(args.max_steps)):
                if is_exploration_episode:
                    action, action_env = _select_exploration_action(
                        trainer=trainer,
                        environment=environment,
                        obs=obs,
                        strategy=args.exploration_strategy,
                        policy_noise_std=args.exploration_policy_noise_std,
                    )
                else:
                    action = trainer.get_action(obs, deterministic=not args.stochastic)
                    if not args.no_env_denormalize:
                        action_env = environment.denormalize(action)
                    else:
                        action_env = action

                next_obs, reward, done, _info = _step_env(environment, action_env)
                states.append(next_obs.copy())
                actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
                rewards.append(float(reward))

                obs = next_obs
                episode_reward += float(reward)
                steps_at_goal += int(reward > args.success_step_reward)
                steps_taken += 1
                if done:
                    break

            trainer.add_episode(states, actions, rewards)
            success = bool(episode_reward > args.success_episode_reward)

            episode_row = {
                "episode_idx": int(episode_idx),
                "episode_seed": int(episode_seed),
                "phase": "exploration" if is_exploration_episode else "training",
                "goal_signature": goal_signature,
                "episode_reward": float(episode_reward),
                "success_episode": int(success),
                "steps_at_goal": int(steps_at_goal),
                "steps_taken": int(steps_taken),
                "online_episodes": len(trainer.online_episodes),
                "total_episodes": trainer.total_episodes,
                "total_transitions": trainer.total_transitions,
            }
            episode_rows.append(episode_row)

            print(
                f"[episode {episode_idx} | {'explore' if is_exploration_episode else 'train'}] "
                f"reward={episode_reward:.4f} "
                f"success={success} steps_at_goal={steps_at_goal} "
                f"steps_taken={steps_taken} total_transitions={trainer.total_transitions}"
            )

            if is_exploration_episode:
                print(
                    f"[train after episode {episode_idx}] skipped "
                    f"(warmup exploration episode "
                    f"{episode_idx + 1}/{args.exploration_episodes})"
                )
            elif trainer.can_train(args.min_transitions_before_training):
                train_metrics = trainer.train_burst(
                    supervised_updates=args.supervised_updates_per_episode,
                    imagination_updates=args.imagination_updates_per_episode,
                )
                train_row = {
                    "episode_idx": int(episode_idx),
                    "phase": "training",
                    "total_episodes": trainer.total_episodes,
                    "total_transitions": trainer.total_transitions,
                    **train_metrics,
                }
                train_rows.append(train_row)
                print(
                    f"[train after episode {episode_idx}] "
                    f"supervised_total={train_metrics.get('supervised_total', float('nan')):.4f} "
                    f"policy_loss={train_metrics.get('policy_loss', float('nan')):.4f} "
                    f"value_loss={train_metrics.get('value_loss', float('nan')):.4f}"
                )
            else:
                print(
                    f"[train after episode {episode_idx}] skipped "
                    f"(need >= {args.min_transitions_before_training} transitions; "
                    f"current max_episode_len={trainer.max_episode_len})"
                )

            if args.save_every_episodes > 0 and (episode_idx + 1) % int(args.save_every_episodes) == 0:
                save_path = checkpoint_dir / f"online_episode_{episode_idx + 1}.pt"
                saved = save_checkpoint(
                    output_path=str(save_path),
                    result=trainer.build_checkpoint_result(),
                    cfg=trainer.cfg,
                    extra={
                        "source_checkpoint": trainer.checkpoint_path,
                        "episode_idx": int(episode_idx),
                        "online_episodes": len(trainer.online_episodes),
                        "total_transitions": trainer.total_transitions,
                    },
                )
                print(f"Saved intermediate checkpoint: {saved}")

    finally:
        if hasattr(environment, "close"):
            try:
                environment.close()
            except Exception:
                pass

    final_checkpoint = checkpoint_dir / "online_final.pt"
    saved = save_checkpoint(
        output_path=str(final_checkpoint),
        result=trainer.build_checkpoint_result(),
        cfg=trainer.cfg,
        extra={
            "source_checkpoint": trainer.checkpoint_path,
            "online_episodes": len(trainer.online_episodes),
            "total_transitions": trainer.total_transitions,
        },
    )

    _save_run_outputs(
        run_dir=run_dir,
        args=args,
        trainer=trainer,
        episode_rows=episode_rows,
        train_rows=train_rows,
    )

    print(f"Saved final checkpoint: {saved}")
    print(f"Saved run outputs: {run_dir}")


if __name__ == "__main__":
    main()
