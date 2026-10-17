
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

ALPHA = 0.5
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
NUM_EPISODES = 10_000
MAX_STEPS = 200
SEED = 42

WINDOW = 100
env = gym.make(
    "FrozenLake-v1",
    map_name="8x8",
    is_slippery=True
)

num_states = env.observation_space.n
num_actions = env.action_space.n

rng = np.random.default_rng(SEED)
env.action_space.seed(SEED)

Q = np.zeros((num_states, num_actions))
rewards = []
epsilon = EPSILON_START

for episode in range(NUM_EPISODES):

    state, info = env.reset(seed=SEED + episode)
    total_reward = 0

    for step in range(MAX_STEPS):

        if rng.random() < epsilon:
            action = int(rng.integers(num_actions))
        else:
            best_actions = np.flatnonzero(
                Q[state] == np.max(Q[state])
            )
            action = int(rng.choice(best_actions))

        next_state, reward, terminated, truncated, info = (
            env.step(action)
        )

        if terminated:
            target = reward
        else:
            target = reward + GAMMA * np.max(Q[next_state])

        Q[state, action] += ALPHA * (target - Q[state, action])
        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    rewards.append(total_reward)
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    if (episode + 1) % 1000 == 0:
        recent_avg = np.mean(rewards[-1000:])
        print(
            f"Episode {episode + 1}: "
            f"Average reward = {recent_avg:.3f}, "
            f"Epsilon = {epsilon:.3f}"
        )

env.close()

#cal moving avg

rewards = np.array(rewards)

moving_average = np.convolve(
    rewards,
    np.ones(WINDOW) / WINDOW,
    mode="valid"
)


print("\nBaseline training completed")
print(f"Total episodes: {NUM_EPISODES}")
print(f"Overall mean reward: {np.mean(rewards):.4f}")
print(f"Last 100 episodes mean reward: {np.mean(rewards[-100:]):.4f}")
print(f"Last 1000 episodes mean reward: {np.mean(rewards[-1000:]):.4f}")

episodes = np.arange(1, NUM_EPISODES + 1)

plt.figure(figsize=(10, 6))

plt.plot(
    episodes,
    rewards,
    alpha=0.25,
    label="Reward per episode"
)

#reward curve
plt.plot(
    np.arange(WINDOW, NUM_EPISODES + 1),
    moving_average,
    linewidth=2,
    label=f"{WINDOW}-episode moving average"
)

plt.xlabel("Training Episode")
plt.ylabel("Total Reward")
plt.title("FrozenLake 8x8: Baseline Q-learning Training")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
