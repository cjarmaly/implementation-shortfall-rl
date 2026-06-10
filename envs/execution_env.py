import gymnasium as gym
import numpy as np

class ExecutionEnv(gym.Env):
    def __init__(self, X, T, eta, gamma, lambda_, 
                ou_theta, ou_mu, ou_sigma, 
                vol_history_length=10):
        super().__init__()
        self.X = X
        self.T = T
        self.eta = eta
        self.gamma = gamma
        self.lambda_ = lambda_
        self.ou_theta = ou_theta
        self.ou_mu = ou_mu
        self.ou_sigma = ou_sigma
        self.vol_history_length = vol_history_length
        
        self.dt = 1 

        # action space is the percentage of inventory to execute
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        # observation space - current inventory, current time, current volatility, volatility history
        low  = np.zeros(3 + vol_history_length, dtype=np.float32)
        high = np.concatenate([
                np.ones(2),                                    # shares, time → [0, 1]
                np.full(1 + vol_history_length, np.inf)        # vol + history → [0, inf)
            ]).astype(np.float32)

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.shares_remaining = self.X
        self.time_remaining = self.T
        self.current_step = 0
        
        self.current_vol = self.ou_mu
        self.vol_history = np.full(self.vol_history_length, self.ou_mu)

        # Build observation vector
        
        # normalize shares and time to [0, 1]
        shares_norm = self.shares_remaining / self.X
        time_norm = (self.T - self.current_step) / self.T

        # concatenate with vol and vol history
        obs = np.concatenate([
                [shares_norm],
                [time_norm],
                [self.current_vol],
                self.vol_history
            ]).astype(np.float32)
        info = {}

        return obs, info

    def step(self, action):
        fraction = action[0]

        shares_to_execute = fraction * self.shares_remaining

        # sample price noise
        price_noise = np.random.normal(0, 1)

        # compute reward (negative IS)
        reward = -(self.eta * shares_to_execute**2 + shares_to_execute * self.current_vol * price_noise)

        # update state
        self.shares_remaining -= shares_to_execute
        self.time_remaining -= self.dt
        self.current_step += 1

        # update volatility and vol history
        Z = np.random.normal(0, 1)
        self.current_vol = self.current_vol + self.ou_theta * (self.ou_mu - self.current_vol) * self.dt + self.ou_sigma * np.sqrt(self.dt) * Z
        self.vol_history = np.roll(self.vol_history, -1)
        self.vol_history[-1] = self.current_vol

        # terminate if time is up or inventory is zero
        terminated = self.time_remaining <= 0 or self.shares_remaining <= 0

        # penalize remaining inventory if not terminated
        if terminated and self.shares_remaining > 0:
            reward -= self.eta * self.shares_remaining**2
            self.shares_remaining = 0

        # buld observation vector
        shares_norm = self.shares_remaining / self.X
        time_norm = (self.T - self.current_step) / self.T
        obs = np.concatenate([
                [shares_norm],
                [time_norm],
                [self.current_vol],
                self.vol_history
            ]).astype(np.float32)
        info = {}

        return obs, reward, terminated, False, info