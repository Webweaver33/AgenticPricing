import os
import sys
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.pricing_env import PricingEnv
from baselines.fixed_price import run_fixed_price_baseline
from agents.q_learning_agent import train_agent


def evaluate_agent(agent, episodes=20):
    env = PricingEnv()
    profits = []

    for _ in range(episodes):
        state, _ = env.reset()
        state_idx = agent.discretize_state(state, env)

        terminated = False
        truncated = False
        total_profit = 0.0

        while not (terminated or truncated):
            action = int(np.argmax(agent.q_table[state_idx]))
            next_state, reward, terminated, truncated, _ = env.step(action)
            state_idx = agent.discretize_state(next_state, env)
            total_profit += reward

        profits.append(total_profit)

    return profits


def main():
    print("Running fixed-price baseline...")
    baseline_results = run_fixed_price_baseline(episodes=20)
    baseline_profit = baseline_results["avg_profit"]

    print("\nTraining learning agent...")
    agent, training_rewards = train_agent(episodes=300)

    print("\nEvaluating trained agent...")
    agent_profits = evaluate_agent(agent, episodes=20)

    avg_agent_profit = np.mean(agent_profits)

    print("\n===== RESULTS =====")
    print(f"Baseline avg profit: {baseline_profit:.2f}")
    print(f"Agent avg profit:    {avg_agent_profit:.2f}")

    # ---- PLOT ----
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(training_rewards, label="Training Profit per Episode", alpha=0.6)
    plt.axhline(
        baseline_profit,
        color="red",
        linestyle="--",
        label="Baseline Avg Profit",
    )
    plt.xlabel("Episode")
    plt.ylabel("Profit")
    plt.title("Agent Learning vs Fixed-Price Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/learning_curve.png")
    plt.close()

    print("\nPlot saved to: plots/learning_curve.png")


if __name__ == "__main__":
    main()
