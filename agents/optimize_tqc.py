import optuna
from sb3_contrib import TQC
from agents.tqc_agent import make_env

def objective(trial):
    # Define hyperparameter search space
    learning_rate = trail.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trail.suggest_categorical("batch_size", [64, 128, 256, 512])
    top_quantiles_to_drop_per_net = trial.suggest_int("top_quantiles_to_drop_per_net", 1, 4)

    env = make_env()

    model = TQC(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=learning_rate,
        batch_size=batch_size,
        buffer_size=50000,
        learning_starts=500,
        top_quantiles_to_drop_per_net=top_quantiles_to_drop_per_net,
    )
    model.learn(total_timesteps=50000)

    # Evaluate over 20 episodes
    total_rewards = 0
    for _ in range(20):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_rewards += reward
            done = terminated or truncated
    return total_rewards / 20.0

def run_study(n_trials=20):
    study = optuna.create_study(drection="maximize")
    study.optimize(objective, n_trails=n_trials)

    print("\nBest trial:")
    print(f"  Value: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")
    
    return study
