
import gymnasium as gym
import numpy as np

LEARNING_RATE = 0.1       #alpha
DISCOUNT_FACTOR = 0.99    #gamma
EPSILON = 1.0              # Initi exploration probability
EPSILON_MIN = 0.01          # Min exploration probability
EPSILON_DECAY = 0.995       #decay after each episode
NUM_EPISODES = 10_000       # num of training episodes
MAX_STEPS = 200             # Max steps per episode
SEED = 42

env = gym.make(
    "FrozenLake-v1",
    map_name="8x8",
    is_slippery=True
)

num_states = env.observation_space.n
num_actions = env.action_space.n

Q = np.zeros((num_states, num_actions))

def choose_action(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    best_actions = np.flatnonzero(Q[state] == np.max(Q[state]))
    return int(np.random.choice(best_actions))


np.random.seed(SEED)
rewards_per_episode = []
epsilon = EPSILON

for episode in range(NUM_EPISODES):
    state, info = env.reset(seed=SEED + episode)
    total_reward = 0

    for step in range(MAX_STEPS):

        # pick and do next action
        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)

        if terminated:
            target = reward
        else:
            target = reward + DISCOUNT_FACTOR * np.max(Q[next_state])

        Q[state, action] += LEARNING_RATE * (target - Q[state, action])
        state = next_state
        total_reward += reward

        #end eps
        if terminated or truncated:
            break
    #totaling rewared
    rewards_per_episode.append(total_reward)
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    #progress display
    if (episode + 1) % 1000 == 0:
        recent_rewards = rewards_per_episode[-1000:]
        average_reward = np.mean(recent_rewards)

        print(
            f"Episode {episode + 1}/{NUM_EPISODES}, "
            f"Average reward: {average_reward:.3f}, "
            f"Epsilon: {epsilon:.3f}"
        )


#display the learned Q-table
print("\nTraining completed!")
print("Q-table shape:", Q.shape)
print("\nLearned Q-table:")
print(np.round(Q, 3))
env.close()
