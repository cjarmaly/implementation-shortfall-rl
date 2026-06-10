import numpy as np
from envs.execution_env import ExecutionEnv
from data.constants import EXECUTION_PARAMS, OU_PARAMS
import torch
from models.gru_policy import GRUFeatureExtractor

def make_env():
    return ExecutionEnv(
        X=EXECUTION_PARAMS["X"],
        T=EXECUTION_PARAMS["T"],
        eta=EXECUTION_PARAMS["eta"],
        gamma=EXECUTION_PARAMS["gamma"],
        lambda_=EXECUTION_PARAMS["lambda_"],
        ou_theta=OU_PARAMS["theta"],
        ou_mu=OU_PARAMS["mu"],
        ou_sigma=OU_PARAMS["sigma"]
    )

def test_obs_shape():
    env = make_env()
    obs, _ = env.reset()
    assert obs.shape == (3 + env.vol_history_length,)

def test_starts_full_inventory():
    env = make_env()
    env.reset()
    assert env.shares_remaining == EXECUTION_PARAMS["X"]

def test_shares_decrease():
    env = make_env()
    env.reset()
    env.step(np.array([0.1]))
    assert env.shares_remaining < EXECUTION_PARAMS["X"]

def test_terminates_at_T():
    env = make_env()
    env.reset()
    terminated = False
    while not terminated:
        _, _, terminated, _, _ = env.step(np.array([0.0]))
    assert env.time_remaining <= 0

def test_vol_evolves():
    env = make_env()
    env.reset()
    initial_vol = env.current_vol
    env.step(np.array([0.0]))
    # vol should have changed via OU step
    assert env.current_vol != initial_vol

def test_gru_extractor_output_shape():
    env = make_env()
    obs, _ = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    extractor = GRUFeatureExtractor(env.observation_space)
    features = extractor(obs_tensor)
    assert features.shape == (1, 3 + 32)