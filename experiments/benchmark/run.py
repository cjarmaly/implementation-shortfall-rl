import numpy as np
import matplotlib.pyplot as plt
from envs.execution_env import ExecutionEnv
from models.almgren_chriss import ac_optimal_trajectory
from agents.tqc_agent import make_env, train_tqc_agent
from stable_baselines3 import PPO
from models.gru_policy import GRUActorCriticPolicy
from data.constants import EXECUTION_PARAMS, OU_PARAMS
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import TQC

def run_episode(env, policy_fn):
    """
    Run one episode using a ccallable policy_fn(obs) --> action.
    Returns total implementation shortfall (positive = cost).
    """

    obs, _ = env.reset()
    total_is = 0.0
    done = False
    while not done:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_is -= reward # negative reward = cost
        done = terminated or truncated
    return total_is

def evaluate(policy_fn, n_episodes=500):
    env = make_env()
    results = [run_episode(env, policy_fn) for _ in range(n_episodes)]
    return np.array(results)

def twap_policy(obs):
    """
    Sell 1/T shares per time step (constant rate)
    """
    # action is fraction of remaining shares to sell this tep
    # TWAP sells evenly, so action = 1 /steps_remaining
    # obs[1] is time_norm = time_remaining / T

    time_norm = obs[1]
    T = EXECUTION_PARAMS["T"]
    steps_remaining = max(1, round(time_norm * T))
    return np.array([1.0 / steps_remaining])

def ac_policy(obs, sigma):
    """
    Recompute AC trajectory at current state and sell the next slice.
    """

    shares_norm = obs[0]
    time_norm = obs[1]
    X = EXECUTION_PARAMS["X"]
    T = EXECUTION_PARAMS["T"]
    shares_remaining = shares_norm * X
    steps_remaining = max(1, round(time_norm * T))

    traj = ac_optimal_trajectory(
        sigma=sigma,
        X=shares_remaining,
        T=steps_remaining,
        eta=EXECUTION_PARAMS["eta"],
        gamma=EXECUTION_PARAMS["gamma"],
        lambda_=EXECUTION_PARAMS["lambda_"],
    )
    shares_to_sell = traj[0] - traj[1]
    fraction = shares_to_sell / shares_remaining if shares_remaining > 0 else 1.0
    return np.array([np.clip(fraction, 0.0, 1.0)])


if __name__ == "__main__":
    env = make_env()   

    # Train PPO with GRU policy
    print("Training PPO+GRU...")

    # Wrap environment in VecNormalize for reward normalization and observation scaling
    ppo_env = DummyVecEnv([make_env])
    ppo_env = VecNormalize(ppo_env, norm_reward=True, norm_obs=False, clip_reward=10.0)

    ppo_model = PPO.load("experiments/benchmark/ppo_model", env=ppo_env) # load model, already trained
    # ppo_model = PPO(GRUActorCriticPolicy, ppo_env, verbose=1, n_steps=512, batch_size=64)
    # ppo_model.learn(total_timesteps=200_000)
    # ppo_model.save("experiments/benchmark/ppo_model")

    # Set VecNormalize to evaluation mode
    ppo_env.training = False
    ppo_env.norm_reward = False

    def evaluate_ppo(model, n_episodes=500):
        env = make_env()  # raw env, no normalization
        results = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            total_is = 0.0
            done = False
            while not done:
                # model.predict expects the VecEnv obs shape, so expand dims
                action, _ = model.predict(obs[np.newaxis, :], deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action[0])
                total_is -= reward
                done = terminated or truncated
            results.append(total_is)
        return np.array(results)


    # Train TQC 
    print("Training TQC...")
    tqc_model = train_tqc_agent(env, total_timesteps=500_000)
    tqc_model.save("experiments/benchmark/tqc_model")

    # Evaluate all four
    print("Evaluating...")

    twap_is = evaluate(lambda obs: twap_policy(obs))
    ac_const_is = evaluate(lambda obs: ac_policy(obs, sigma=OU_PARAMS["mu"]))
    ac_roll_is  = evaluate(lambda obs: ac_policy(obs, sigma=obs[2]))  # live vol from obs
    ppo_is = evaluate_ppo(ppo_model)
    tqc_is = evaluate(lambda obs: tqc_model.predict(obs, deterministic=True)[0])

    # Print summary
    for name, arr in [("TWAP", twap_is), ("AC constant vol", ac_const_is),
                      ("AC rolling vol", ac_roll_is), ("PPO+GRU", ppo_is), ("TQC", tqc_is)]:
        print(f"{name:20s}  mean IS: {arr.mean():.2f}  std: {arr.std():.2f}  CVaR95: {np.percentile(arr, 95):.2f}")

    # Plot IS distributions 
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, arr in [("TWAP", twap_is), ("AC constant vol", ac_const_is),
                      ("AC rolling vol", ac_roll_is), ("PPO+GRU", ppo_is), ("TQC", tqc_is)]:
        ax.hist(arr, bins=50, alpha=0.5, label=name, density=True)
    ax.set_xlabel("Implementation Shortfall")
    ax.set_ylabel("Density")
    ax.set_title("IS Distribution: Four-Way Benchmark")
    ax.legend()
    plt.tight_layout()
    plt.savefig("experiments/benchmark/is_distributions.png", dpi=150)
    plt.show()
    print("Saved: experiments/benchmark/is_distributions.png")