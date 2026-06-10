from sb3_contrib import TQC
from models.gru_policy import GRUActorCriticPolicy
from envs.execution_env import ExecutionEnv
from data.constants import EXECUTION_PARAMS, OU_PARAMS

"""
Training a TQC (Truncated Quantile Critics) agent for the execution problem.

The agent is SAC but with distribution critics that truncate the top quantiles to reduce
overestimation of the value funciton. It typically outperforms SAC on continuous control tasks.
"""

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

def train_tqc_agent(env, total_timesteps=200000):
    model = TQC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=100000,
        learning_starts=1000,
        top_quantiles_to_drop_per_net=2,
    )
    model.learn(total_timesteps=total_timesteps)
    return model
