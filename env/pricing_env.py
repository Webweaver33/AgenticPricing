import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PricingEnv(gym.Env):
    """
    Pricing environment where an agent selects prices to maximize profit.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        initial_inventory=100,
        base_demand=20.0,
        price_elasticity=1.5,
        competitor_price_mean=10.0,
        competitor_price_std=1.0,
        max_steps=50,
        unit_cost=4.0,
    ):
        super().__init__()

        self.initial_inventory = initial_inventory
        self.base_demand = base_demand
        self.price_elasticity = price_elasticity
        self.competitor_price_mean = competitor_price_mean
        self.competitor_price_std = competitor_price_std
        self.max_steps = max_steps
        self.unit_cost = unit_cost

        self.price_levels = np.array([6, 7, 8, 9, 10, 11, 12], dtype=float)

        self.action_space = spaces.Discrete(len(self.price_levels))

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([np.inf, np.inf, np.inf, self.max_steps]),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.inventory = self.initial_inventory
        self.current_step = 0
        self.current_price = float(self.price_levels.mean())
        self.competitor_price = self._sample_competitor_price()

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        chosen_price = self.price_levels[action]
        self.current_price = chosen_price

        demand = self._calculate_demand(
            price=chosen_price,
            competitor_price=self.competitor_price,
        )

        sales = min(demand, self.inventory)
        revenue = sales * chosen_price
        cost = sales * self.unit_cost
        profit = revenue - cost

        self.inventory -= sales
        self.current_step += 1
        self.competitor_price = self._sample_competitor_price()

        terminated = self.inventory <= 0
        truncated = self.current_step >= self.max_steps

        observation = self._get_observation()
        reward = profit
        info = {
            "sales": sales,
            "revenue": revenue,
            "profit": profit,
            "inventory": self.inventory,
        }

        return observation, reward, terminated, truncated, info

    def _calculate_demand(self, price, competitor_price):
        relative_price = price / competitor_price
        demand = self.base_demand * np.exp(-self.price_elasticity * relative_price)
        return max(0.0, demand)

    def _sample_competitor_price(self):
        return max(
            1.0,
            np.random.normal(self.competitor_price_mean, self.competitor_price_std),
        )

    def _get_observation(self):
        return np.array(
            [
                self.current_price,
                self.competitor_price,
                self.inventory,
                self.current_step,
            ],
            dtype=np.float32,
        )
