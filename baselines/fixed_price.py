import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.pricing_env import PricingEnv


def run_fixed_price_baseline(
    fixed_price_index=3,
    episodes=10,
):
    total_profit = 0.0
    stockouts = 0

    for _ in range(episodes):
        env = PricingEnv()
        obs, _ = env.reset()

        terminated = False
        truncated = False

        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(fixed_price_index)
            total_profit += reward

        if info["inventory"] <= 0:
            stockouts += 1

    return {
        "strategy": "Fixed Price",
        "price": env.price_levels[fixed_price_index],
        "avg_profit": total_profit / episodes,
        "stockout_rate": stockouts / episodes,
    }


if __name__ == "__main__":
    results = run_fixed_price_baseline()
    for k, v in results.items():
        print(f"{k}: {v}")
