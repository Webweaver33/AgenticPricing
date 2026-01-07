import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.pricing_env import PricingEnv


class QLearningAgent:
    def __init__(
        self,
        state_bins,
        n_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    ):
        self.state_bins = state_bins
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = np.zeros(state_bins + (n_actions,))

    def discretize_state(self, state, env):
        price, competitor_price, inventory, step = state

        price_bin = np.digitize(price, env.price_levels) - 1
        competitor_bin = int(np.clip(competitor_price // 2, 0, self.state_bins[1] - 1))
        inventory_bin = int(np.clip(inventory // 10, 0, self.state_bins[2] - 1))
        step_bin = int(np.clip(step, 0, self.state_bins[3] - 1))

        return (price_bin, competitor_bin, inventory_bin, step_bin)

    def choose_action(self, state_idx):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state_idx]))

    def update(self, state_idx, action, reward, next_state_idx, done):
        best_next_q = np.max(self.q_table[next_state_idx])
        target = reward + self.gamma * best_next_q * (not done)
        self.q_table[state_idx][action] += self.lr * (
            target - self.q_table[state_idx][action]
        )

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent(
    episodes=300,
):
    env = PricingEnv()

    state_bins = (7, 6, 11, env.max_steps)
    agent = QLearningAgent(
        state_bins=state_bins,
        n_actions=len(env.price_levels),
    )

    episode_rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        state_idx = agent.discretize_state(state, env)

        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action = agent.choose_action(state_idx)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state_idx = agent.discretize_state(next_state, env)

            agent.update(
                state_idx,
                action,
                reward,
                next_state_idx,
                terminated or truncated,
            )

            state_idx = next_state_idx
            total_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if (ep + 1) % 50 == 0:
            print(
                f"Episode {ep+1} | "
                f"Avg Profit (last 50): {np.mean(episode_rewards[-50:]):.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    return agent, episode_rewards


if __name__ == "__main__":
    agent, rewards = train_agent()
    print("Training complete.")
