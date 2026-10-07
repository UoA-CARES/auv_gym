"""
Fixed Dreamer4 with improvements for your reward structure:
- Rewards range: -1 to +5
- Dense distance-based rewards (mostly negative when far from goal)
- Sparse +5 bonus at goal

Key fixes:
1. Better reward model architecture
2. Don't normalize rewards (breaks the scale!)
3. Better value function training
4. Reward clipping for stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import argparse
import math
from typing import Optional
from dataclasses import dataclass
from collections import deque
from pathlib import Path

# Copy all the base classes from dreamer4_standalone.py
# (RMSNorm, RotaryEmbedding, SwiGLU, CausalSelfAttention, etc.)
# I'll just show the key changes below

@dataclass
class Trajectory:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray


class RunningNormalizer:
    def __init__(self, shape, epsilon=1e-8, clip=10.0):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
        self.epsilon = epsilon
        self.clip = clip
    
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean += delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count
    
    def normalize(self, x):
        normalized = (x - self.mean) / (np.sqrt(self.var) + self.epsilon)
        return np.clip(normalized, -self.clip, self.clip)


# ============================================================================
# IMPROVED REWARD MODEL - Deeper and more expressive
# ============================================================================

class ImprovedRewardModel(nn.Module):
    """
    Deeper reward model for learning complex reward functions.
    Critical for tasks with sparse bonuses!
    """
    def __init__(self, obs_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, obs):
        return self.net(obs).squeeze(-1)


# ============================================================================
# KEY CHANGES TO AGENT
# ============================================================================

class ImprovedDreamer4Agent:
    """
    Improved agent with fixes for your reward structure
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        reward_hidden_dim: int = 512,  # Larger reward model
        n_layers: int = 4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        normalize_obs: bool = True,
        normalize_rewards: bool = False,  # DON'T normalize rewards!
        reward_clip: tuple = (-1.5, 6.0)  # Clip rewards for stability
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.normalize_obs = normalize_obs
        self.normalize_rewards = normalize_rewards
        self.reward_clip = reward_clip
        
        print(f"\n{'='*60}")
        print(f"Initializing Improved Dreamer4Agent on: {device.upper()}")
        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Reward range: {reward_clip}")
        print(f"Normalization: obs={normalize_obs}, rewards={normalize_rewards}")
        print(f"{'='*60}\n")
        
        # Import base models
        from dreamer4_standalone import WorldModel, PolicyNetwork, ValueNetwork, ReplayBuffer
        
        # Models
        self.world_model = WorldModel(obs_dim, action_dim, hidden_dim, n_layers).to(device)
        self.policy = PolicyNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.value = ValueNetwork(obs_dim, hidden_dim).to(device)
        
        # IMPROVED: Larger reward model
        self.reward = ImprovedRewardModel(obs_dim, reward_hidden_dim).to(device)
        
        # Normalizers
        self.obs_normalizer = RunningNormalizer(obs_dim) if normalize_obs else None
        self.reward_normalizer = RunningNormalizer(1) if normalize_rewards else None
        
        # IMPROVED: Better learning rates for reward model
        self.world_optimizer = torch.optim.AdamW(self.world_model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.policy_optimizer = torch.optim.AdamW(self.policy.parameters(), lr=3e-5, weight_decay=1e-4)
        self.value_optimizer = torch.optim.AdamW(self.value.parameters(), lr=3e-4, weight_decay=1e-4)  # Higher LR
        self.reward_optimizer = torch.optim.AdamW(self.reward.parameters(), lr=3e-4, weight_decay=1e-5)  # Higher LR, less decay
        
        self.replay_buffer = ReplayBuffer()
        
        self.training_stats = {
            'world_model_losses': [],
            'reward_losses': [],
            'reward_correlations': [],  # Track reward prediction quality
            'policy_losses': [],
            'value_losses': [],
            'avg_rewards': [],
            'max_rewards': []
        }
    
    def normalize_observations(self, obs):
        if self.obs_normalizer is None:
            return obs
        
        is_tensor = isinstance(obs, torch.Tensor)
        device = obs.device if is_tensor else None
        obs_np = obs.cpu().numpy() if is_tensor else obs
        
        if obs_np.ndim == 3:
            flat_obs = obs_np.reshape(-1, obs_np.shape[-1])
            self.obs_normalizer.update(flat_obs)
            normalized = self.obs_normalizer.normalize(obs_np)
        else:
            self.obs_normalizer.update(obs_np)
            normalized = self.obs_normalizer.normalize(obs_np)
        
        return torch.FloatTensor(normalized).to(device) if is_tensor else normalized
    
    def clip_rewards(self, rewards):
        """Clip rewards to reasonable range for stability"""
        if isinstance(rewards, torch.Tensor):
            return torch.clamp(rewards, self.reward_clip[0], self.reward_clip[1])
        else:
            return np.clip(rewards, self.reward_clip[0], self.reward_clip[1])
    
    def compute_world_model_loss(self, obs, actions):
        B, T, _ = obs.shape
        d_options = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64]
        d = torch.tensor([d_options[np.random.randint(len(d_options))] for _ in range(B * T)], device=self.device).view(B, T)
        tau = torch.rand(B, T, device=self.device)
        
        noise = torch.randn_like(obs)
        obs_corrupted = (1 - tau.unsqueeze(-1)) * noise + tau.unsqueeze(-1) * obs
        
        tau_disc = (tau * 99).long().clamp(0, 99)
        d_disc = (torch.log2(1.0 / d.clamp(min=1/64)) * 2).long().clamp(0, 19)
        
        pred_obs = self.world_model(obs_corrupted, actions, tau_disc, d_disc)
        
        weight = 0.9 * tau.unsqueeze(-1) + 0.1
        loss = weight * F.mse_loss(pred_obs, obs, reduction='none')
        
        return loss.mean()
    
    def compute_lambda_returns(self, rewards, values, gamma=0.99, lambda_=0.95):
        B, T = rewards.shape
        returns = torch.zeros_like(rewards)
        
        last_value = values[:, -1]
        for t in reversed(range(T)):
            if t == T - 1:
                returns[:, t] = rewards[:, t] + gamma * last_value
            else:
                returns[:, t] = rewards[:, t] + gamma * ((1 - lambda_) * values[:, t + 1] + lambda_ * returns[:, t + 1])
        
        return returns
    
    def train_world_model(self, num_steps: int = 10000, batch_size: int = 32, seq_len: int = 16):
        """Phase 1: Train world model with EMPHASIS on reward prediction"""
        print("Phase 1: Training world model...")
        print(f"Training for {num_steps} steps (longer for better reward learning)")
        
        for step in range(num_steps):
            obs, actions, rewards = self.replay_buffer.sample(batch_size, seq_len)
            
            obs_norm = self.normalize_observations(obs)
            obs_norm = obs_norm.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            
            # Clip rewards for stability
            rewards = self.clip_rewards(rewards)
            
            # World model loss
            wm_loss = self.compute_world_model_loss(obs_norm, actions)
            
            self.world_optimizer.zero_grad()
            wm_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
            self.world_optimizer.step()
            
            # Reward model loss with HIGHER WEIGHT
            B, T, _ = obs_norm.shape
            obs_flat = obs_norm.view(B * T, -1)
            rewards_flat = rewards.view(B * T)
            
            pred_rewards = self.reward(obs_flat)
            reward_loss = F.mse_loss(pred_rewards, rewards_flat)
            
            # IMPORTANT: Train reward model MORE
            self.reward_optimizer.zero_grad()
            reward_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward.parameters(), 1.0)
            self.reward_optimizer.step()
            
            # Track statistics
            self.training_stats['world_model_losses'].append(wm_loss.item())
            self.training_stats['reward_losses'].append(reward_loss.item())
            
            # Track reward prediction quality
            if step % 100 == 0:
                with torch.no_grad():
                    correlation = np.corrcoef(
                        rewards_flat.cpu().numpy(),
                        pred_rewards.cpu().numpy()
                    )[0, 1]
                    self.training_stats['reward_correlations'].append(correlation)
                
                print(f"Step {step}: WM Loss = {wm_loss.item():.4f}, "
                      f"Reward Loss = {reward_loss.item():.4f}, "
                      f"Reward Corr = {correlation:.3f}")
                
                if correlation < 0.5 and step > 1000:
                    print(f"  ⚠️ WARNING: Low reward correlation! Reward model struggling.")
    
    def train_policy_in_imagination(
        self, 
        num_steps: int = 5000, 
        batch_size: int = 32, 
        horizon: int = 20,  # Longer horizon to see goal bonus
        context_len: int = 5,
        gamma: float = 0.99,
        lambda_: float = 0.95
    ):
        """Phase 2: Improved imagination training"""
        print("Phase 2: Training policy in imagination...")
        print(f"Horizon: {horizon} (longer to see goal bonuses)")
        
        for step in range(num_steps):
            obs_ctx, _, _ = self.replay_buffer.sample(batch_size, context_len)
            obs_ctx = self.normalize_observations(obs_ctx).to(self.device)
            
            with torch.no_grad():
                curr_obs = obs_ctx[:, -1]
                imagined_obs_list = [obs_ctx]
                imagined_actions_list = []
                
                for t in range(horizon):
                    action = self.policy.get_action(curr_obs, deterministic=False)
                    imagined_actions_list.append(action)
                    
                    obs_input = curr_obs.unsqueeze(1)
                    action_input = action.unsqueeze(1)
                    tau_seq = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
                    d_seq = torch.full((batch_size, 1), 10, dtype=torch.long, device=self.device)
                    
                    next_obs = self.world_model(obs_input, action_input, tau_seq, d_seq)[:, 0]
                    imagined_obs_list.append(next_obs.unsqueeze(1))
                    curr_obs = next_obs
                
                imagined_obs = torch.cat(imagined_obs_list, dim=1)
            
            # Predict rewards and values
            B, T, _ = imagined_obs.shape
            obs_flat = imagined_obs.view(B * T, -1)
            
            rewards = self.reward(obs_flat).view(B, T)
            rewards = self.clip_rewards(rewards)  # Clip for stability
            
            values = self.value(obs_flat).view(B, T)
            
            # Compute returns
            lambda_returns = self.compute_lambda_returns(rewards, values, gamma, lambda_)
            advantages = lambda_returns - values
            
            # Normalize advantages per batch (not globally)
            advantages_std = advantages.std() + 1e-8
            advantages_norm = advantages / advantages_std
            
            # Update value network
            value_loss = F.mse_loss(values, lambda_returns.detach())
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
            self.value_optimizer.step()
            
            # Update policy
            policy_obs = imagined_obs[:, context_len:].detach()
            B, H, _ = policy_obs.shape
            policy_actions = self.policy(policy_obs.reshape(B * H, -1)).reshape(B, H, -1)
            policy_advantages = advantages_norm[:, context_len:].detach()
            
            # Policy gradient loss
            policy_loss = -torch.mean(policy_advantages.unsqueeze(-1) * policy_actions)
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()
            
            # Track statistics
            self.training_stats['policy_losses'].append(policy_loss.item())
            self.training_stats['value_losses'].append(value_loss.item())
            self.training_stats['avg_rewards'].append(rewards.mean().item())
            self.training_stats['max_rewards'].append(rewards.max().item())
            
            if step % 100 == 0:
                print(f"Step {step}: Policy Loss = {policy_loss.item():.4f}, "
                      f"Value Loss = {value_loss.item():.4f}, "
                      f"Avg Reward = {rewards.mean().item():.4f}, "
                      f"Max Reward = {rewards.max().item():.4f}, "
                      f"Avg Advantage = {advantages.mean().item():.4f}")
                
                # Check if seeing any goal bonuses
                if step > 500 and rewards.max().item() < 1.0:
                    print(f"  ⚠️ Not seeing high rewards (max={rewards.max().item():.2f}). "
                          f"Policy may not be reaching goals in imagination.")
    
    @torch.no_grad()
    def get_action(self, obs, deterministic=False):
        obs_norm = self.normalize_observations(obs)
        obs_tensor = torch.FloatTensor(obs_norm).unsqueeze(0).to(self.device)
        action = self.policy.get_action(obs_tensor, deterministic)
        return action.cpu().numpy()[0] #.flatten()? or .squeeze()?
    
    def save(self, path: str):
        save_dict = {
            'world_model': self.world_model.state_dict(),
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'reward': self.reward.state_dict(),
            'training_stats': self.training_stats
        }
        
        if self.obs_normalizer is not None:
            save_dict['obs_normalizer'] = {
                'mean': self.obs_normalizer.mean,
                'var': self.obs_normalizer.var,
                'count': self.obs_normalizer.count
            }
        
        torch.save(save_dict, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.world_model.load_state_dict(checkpoint['world_model'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])
        self.reward.load_state_dict(checkpoint['reward'])
        
        if 'obs_normalizer' in checkpoint and self.obs_normalizer is not None:
            self.obs_normalizer.mean = checkpoint['obs_normalizer']['mean']
            self.obs_normalizer.var = checkpoint['obs_normalizer']['var']
            self.obs_normalizer.count = checkpoint['obs_normalizer']['count']
        
        print(f"Model loaded from {path}")
    
    def plot_training_curves(self, save_path='training_curves.png'):
        """Plot detailed training curves"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            
            # WM Loss
            axes[0, 0].plot(self.training_stats['world_model_losses'])
            axes[0, 0].set_title('World Model Loss')
            axes[0, 0].set_yscale('log')
            axes[0, 0].set_xlabel('Step')
            
            # Reward Loss
            axes[0, 1].plot(self.training_stats['reward_losses'])
            axes[0, 1].set_title('Reward Model Loss')
            axes[0, 1].set_xlabel('Step')
            
            # Reward Correlation
            if len(self.training_stats['reward_correlations']) > 0:
                axes[0, 2].plot(self.training_stats['reward_correlations'])
                axes[0, 2].set_title('Reward Prediction Correlation')
                axes[0, 2].axhline(y=0.7, color='g', linestyle='--', label='Good (>0.7)')
                axes[0, 2].axhline(y=0.5, color='orange', linestyle='--', label='OK (>0.5)')
                axes[0, 2].axhline(y=0.3, color='r', linestyle='--', label='Bad (<0.3)')
                axes[0, 2].legend()
                axes[0, 2].set_xlabel('Step (x100)')
            
            # Policy Loss
            if len(self.training_stats['policy_losses']) > 0:
                axes[1, 0].plot(self.training_stats['policy_losses'])
                axes[1, 0].set_title('Policy Loss')
                axes[1, 0].set_xlabel('Step')
            
            # Value Loss
            if len(self.training_stats['value_losses']) > 0:
                axes[1, 1].plot(self.training_stats['value_losses'])
                axes[1, 1].set_title('Value Loss')
                axes[1, 1].set_xlabel('Step')
            
            # Rewards
            if len(self.training_stats['avg_rewards']) > 0:
                axes[1, 2].plot(self.training_stats['avg_rewards'], label='Avg', alpha=0.7)
                axes[1, 2].plot(self.training_stats['max_rewards'], label='Max', alpha=0.7)
                axes[1, 2].axhline(y=5.0, color='g', linestyle='--', label='Goal Bonus (+5)')
                axes[1, 2].axhline(y=0.0, color='gray', linestyle='-', alpha=0.3)
                axes[1, 2].set_title('Imagined Rewards')
                axes[1, 2].legend()
                axes[1, 2].set_xlabel('Step')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            print(f"Training curves saved to {save_path}")
            plt.close()
        except ImportError:
            print("matplotlib not installed, skipping plots")


# ============================================================================
# Training Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--obs_dim', type=int, required=True)
    parser.add_argument('--action_dim', type=int, required=True)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--reward_hidden_dim', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--pretrain_steps', type=int, default=10000)  # More steps!
    parser.add_argument('--imagination_steps', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--horizon', type=int, default=20)  # Longer horizon
    parser.add_argument('--save_path', type=str, default='./checkpoints/dreamer4_fixed.pt')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    with open(args.data_path, 'rb') as f:
        trajectories = pickle.load(f)
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Analyze rewards
    all_rewards = np.concatenate([t.rewards for t in trajectories])
    print(f"\n📊 Reward Statistics:")
    print(f"  Mean: {all_rewards.mean():.3f}")
    print(f"  Std: {all_rewards.std():.3f}")
    print(f"  Range: [{all_rewards.min():.3f}, {all_rewards.max():.3f}]")
    print(f"  Goal bonuses (+5): {np.sum(all_rewards > 4.5)}")
    
    # Create agent
    agent = ImprovedDreamer4Agent(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        reward_hidden_dim=args.reward_hidden_dim,
        n_layers=args.n_layers,
        normalize_obs=True,
        normalize_rewards=False,  # Don't normalize!
        reward_clip=(-1.5, 6.0)
    )
    
    # Add data
    for traj in trajectories:
        agent.replay_buffer.add(traj)
    print(f"Added {len(agent.replay_buffer)} trajectories to replay buffer\n")
    
    # Train
    agent.train_world_model(num_steps=args.pretrain_steps, batch_size=args.batch_size)
    agent.train_policy_in_imagination(
        num_steps=args.imagination_steps, 
        batch_size=args.batch_size,
        horizon=args.horizon
    )
    
    # Save
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.save_path)
    agent.plot_training_curves(args.save_path.replace('.pt', '_curves.png'))
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nFinal Statistics:")
    print(f"  Best reward correlation: {max(agent.training_stats['reward_correlations']):.3f}")
    print(f"  Max imagined reward: {max(agent.training_stats['max_rewards']):.3f}")
    print(f"  Final avg reward: {agent.training_stats['avg_rewards'][-1]:.3f}")


if __name__ == "__main__":
    main()
