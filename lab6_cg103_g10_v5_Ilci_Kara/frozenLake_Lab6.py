import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import datetime


# Q-table check and user prompt
import sys

# Initialize slippery mode variables with default values
train_slippery_mode = False
slippery_mode = False

TEST_EPISODES = 100

print("\n🔍 Checking for existing Q-table...")
if os.path.exists("q_table_nonslippery.npy") or os.path.exists("q_table_slippery.npy"):
    # Slippery mode prompt
    slip_input = (
        input("Do you want the environment to be slippery? (y/n): ").strip().lower()
    )
    slippery_mode = True if slip_input == "y" else False

    # Training slippery mode prompt
    train_slip_input = (
        input("Do you want to TRAIN in slippery mode? (y/n): ").strip().lower()
    )
    train_slippery_mode = True if train_slip_input == "y" else False

    response = (
        input(
            "A saved Q-table was found. Do you want to skip training and use it for rendering only? (y/n): "
        )
        .strip()
        .lower()
    )

    if response == "y" and train_slippery_mode:
        q_table = np.load("q_table_slippery.npy")
        print("✅ Loaded saved Q-table.")
        render_only = True

    elif response == "y" and not train_slippery_mode:
        q_table = np.load("q_table_nonslippery.npy")
        print("✅ Loaded saved Q-table.")
        render_only = True
    else:
        print("⏳ Proceeding with training.")
        render_only = False
else:
    print("⚠️ No saved Q-table found. Proceeding with training.")
    render_only = False


if render_only:
    # Load correct Q-table based on test environment
    q_filename = "q_table_slippery.npy" if slippery_mode else "q_table_nonslippery.npy"
    q_table = np.load(q_filename)
    print("✅ Loaded saved Q-table.")
    # Directly test and render using the loaded Q-table
    print("\nTesting trained agent (no render)...")
    successes = 0
    test_episodes = TEST_EPISODES

    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=slippery_mode)
    max_steps_per_episode = 200
    for episode in range(test_episodes):
        state, _ = env.reset()
        for _ in range(max_steps_per_episode):
            action = np.argmax(q_table[state])
            state, reward, done, truncated, info = env.step(action)
            if done:
                if reward == 1:
                    successes += 1
                break

    print(f"Success rate of trained agent: {successes/test_episodes:.2%}")

    print("\nRunning trained agent with render...\n")
    env_render = gym.make(
        "FrozenLake-v1", map_name="8x8", is_slippery=slippery_mode, render_mode="human"
    )
    state, _ = env_render.reset()

    for step in range(max_steps_per_episode):
        action = np.argmax(q_table[state])
        state, reward, done, truncated, info = env_render.step(action)
        if done:
            if reward == 1:
                print(f"✅ Goal reached! Step: {step}")
            else:
                print(f"❌ Goal not reached. Step: {step}")
            break

    # Exit early if we only wanted to render
    raise SystemExit

#
# Environment creation (no render - for training/testing)
# (If render_only, we exited above, so create here)
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=train_slippery_mode)

#
# Q-table initialization
state_space_size = env.observation_space.n
action_space_size = env.action_space.n
q_table = np.zeros((state_space_size, action_space_size))

#
# Hyperparameters (adjusted for slippery mode)
if train_slippery_mode:
    num_episodes = 100000
    learning_rate = 0.6
    exploration_decay_rate = 0.00001
else:
    num_episodes = 20000
    learning_rate = 0.8
    exploration_decay_rate = 0.00005

max_steps_per_episode = 200
discount_rate = 0.99

exploration_rate = 1.0
print(f"[DEBUG] Starting exploration_rate: {exploration_rate}")
max_exploration_rate = 1.0
min_exploration_rate = 0.01

#
# Training statistics
rewards_all_episodes = []

#
# Q-learning algorithm
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
        # Epsilon-greedy action selection
        if np.random.rand() > exploration_rate:
            action = np.argmax(q_table[state])
        else:
            action = env.action_space.sample()

        new_state, reward, done, truncated, info = env.step(action)

        # Punish falling into a hole in slippery mode
        if train_slippery_mode and reward == 0 and done:
            reward = -1  # punish falling into a hole

        # Q-table update
        q_table[state, action] = q_table[state, action] * (
            1 - learning_rate
        ) + learning_rate * (reward + discount_rate * np.max(q_table[new_state]))

        state = new_state
        # Small penalty at each step (encourages shortest path)
        reward -= 0.01
        rewards_current_episode += reward

        if done:
            break

    # Decrease exploration rate
    exploration_rate = min_exploration_rate + (
        max_exploration_rate - min_exploration_rate
    ) * np.exp(-exploration_decay_rate * episode)

    # Learning rate decay for slippery mode
    if train_slippery_mode:
        learning_rate = max(0.1, learning_rate * 0.99995)

    if episode % 1000 == 0:
        print(f"[DEBUG] Exploration rate at episode {episode}: {exploration_rate:.4f}")

    if episode == 1000:
        early_rewards = np.sum(rewards_all_episodes[:1000])
        print(f"[DEBUG] Total reward in first 1000 episodes: {early_rewards}")

    rewards_all_episodes.append(rewards_current_episode)
    # Save reward per episode to a file for later analysis
    with open("reward_log.txt", "a") as reward_log:
        reward_log.write(f"{rewards_current_episode}\n")

    # Intermediate output
    if episode % 1000 == 0 and episode != 0:
        avg_reward = np.mean(rewards_all_episodes[episode - 1000 : episode])
        print(
            f"Episodes {episode-1000}-{episode}: average reward = {avg_reward:.4f}, exploration_rate = {exploration_rate:.4f}"
        )


# Learned Q-table
print("\nLearned Q-table:")
print(q_table)
q_filename = (
    "q_table_slippery.npy" if train_slippery_mode else "q_table_nonslippery.npy"
)
np.save(q_filename, q_table)
print(f"✅ Q-table successfully saved to '{q_filename}'.")

# --------------------------
# TEST: Success rate with trained agent
# --------------------------
print("\nTesting trained agent (no render)...")
successes = 0
test_episodes = TEST_EPISODES

for episode in range(test_episodes):
    state, _ = env.reset()
    for _ in range(max_steps_per_episode):
        action = np.argmax(q_table[state])  # Always pick the best action
        state, reward, done, truncated, info = env.step(action)
        if done:
            if reward == 1:
                successes += 1
            break

print(f"Success rate of trained agent: {successes/test_episodes:.2%}")
log_message = f"{datetime.datetime.now()}: Success rate of trained agent: {successes/test_episodes:.2%}\n"
with open("training_log.txt", "a") as log_file:
    log_file.write(log_message)

# --------------------------
# APPLICATION: Run agent with render
# --------------------------
print("\nRunning trained agent with render...\n")
env_render = gym.make(
    "FrozenLake-v1", map_name="8x8", is_slippery=slippery_mode, render_mode="human"
)
state, _ = env_render.reset()

for step in range(max_steps_per_episode):
    action = np.argmax(q_table[state])
    state, reward, done, truncated, info = env_render.step(action)
    if done:
        if reward == 1:
            print(f"✅ Goal reached! Step: {step}")
        else:
            print(f"❌ Goal not reached. Step: {step}")
        break
else:
    print(f"⛔ Maximum step limit reached. Step: {step + 1}")

# --------------------------
# Graphical analysis
# --------------------------
# Mean reward per episode group (every 1000 episodes)
group_size = 1000
num_groups = len(rewards_all_episodes) // group_size
grouped_avg_rewards = [
    np.mean(rewards_all_episodes[i * group_size : (i + 1) * group_size])
    for i in range(num_groups)
]

plt.figure(figsize=(12, 5))
plt.plot(
    np.arange(num_groups) * group_size, grouped_avg_rewards, marker="o", color="purple"
)
plt.xlabel("Episode")
plt.ylabel(f"Avg Reward per {group_size} Episodes")
plt.title(f"Grouped Average Reward (Group size = {group_size})")
plt.grid(True)
plt.tight_layout()
plt.savefig("avg_reward_per_1000_episodes.png")
plt.show()

# Reward per episode
plt.figure(figsize=(12, 5))
plt.plot(rewards_all_episodes, color="blue")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward per Episode")
plt.grid(True)
plt.tight_layout()
plt.savefig("reward_per_episode.png")
plt.show()

# Moving average of reward (window=100)
window = 100
moving_avg = np.convolve(rewards_all_episodes, np.ones(window) / window, mode="valid")
plt.figure(figsize=(12, 5))
plt.plot(moving_avg, color="green")
plt.xlabel("Episode")
plt.ylabel(f"Average Reward (window={window})")
plt.title("Moving Average of Rewards")
plt.grid(True)
plt.tight_layout()
plt.savefig("moving_average_rewards.png")
plt.show()
