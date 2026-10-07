from tools.dreamer4_reward_fix import ImprovedDreamer4Agent
from environments.environment_factory import EnvironmentFactory

# load agent
agent = ImprovedDreamer4Agent(obs_dim=6, action_dim=1)
agent.load('./checkpoints/dreamer4_fixed.pt')

# create environment
env_factory = EnvironmentFactory()
environment = env_factory.create_environment("boxfish", "FiveMarkerSpin")

# check environment
print("Resetting Environment")
state = environment.reset()
print(f"State: {state}")

observation_size = len(state)
# action_num = len(auv_config.control_actions) # starting with only xyz, will parameterize later to include roll, pitch, yaw, grabber
print(
    f"Observation Space: {observation_size}"
)

# Run 10 episodes in your real environment
success_count = 0
for ep in range(10):
    obs = environment.reset()
    episode_reward = 0
    ep_success_count = 0

    for step in range(20):  # Your max steps
        action = agent.get_action(obs, deterministic=True)
        action_env = environment.denormalize(action)
        obs, reward, done, info = environment.step(action_env)

        ep_success_count += 1 if reward > 4 else 0
        episode_reward += reward
    
    reached_goal = episode_reward > 4.0
    success_count += reached_goal
    print(f"Episode {ep}: Reward={episode_reward:.2f}, Success={reached_goal}, Steps at goal={ep_success_count}")

print(f"\nSuccess Rate: {success_count}/10 = {success_count*10}%")