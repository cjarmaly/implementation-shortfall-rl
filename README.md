# Optimal Execution with Stochastic Volatility and Reinforcement Learning

You need to sell 10,000 shares of SPY in a certain number of time steps (we arbitrarily chose 26). Sell too fast and you move the market against yourself. Sell too slow and you accumulate risk. The Almgren-Chriss (AC) model gives an analytical solution to this tradeoff, but it arrogantly assumes volatility is constant. Volatitly, in fact, is not constant, leading us to question if we can beat the AC model with additional information about the randomness of volatility.

The key here is calibrating a real intraday volatility process from SPY 15-minute bar data, fitting an Ornstein-Uhlenbeck model to it, and then asking whether RL agents that can see the current volatility and its recent history can outperform both the Time-Weighted Average Price (TWAP) and the analytical AC benchmark. The answer, after a lot of debugging and excessive agent training, is yes. 

---

## What I Built

I started with data. Using `yfinance`, I pulled 60 days of SPY intraday data at 15-minute intervals and computed realized volatility using a rolling window. The raw series has the U-shape you'd expect — elevated at open, quiet at midday, spiking at close. I then fit an OU process to it using closed-form OLS regression: regress X[t+1] on X[t], recover θ, μ, and σ analytically from the coefficients and residuals. MLE was the first approach and it failed immediately — sigma estimates on the order of 1e-29. OLS worked on the first try. Calibrated parameters: θ=0.114, μ=0.062, σ=0.0045.

With a calibrated vol process in hand, the next step was the analytical baseline. The Almgren-Chriss model takes a risk-aversion parameter λ, market impact coefficients η and γ, and a volatility estimate, and returns the optimal liquidation trajectory in closed form via a sinh expression. I implemented both the trajectory and the expected implementation shortfall formula, verified them against known properties (TWAP as λ→0, front-loading as λ increases), and wrote pytest tests before touching the environment.

The Gymnasium execution environment wraps everything into a learning problem. The agent sells shares over T=26 steps. Its observation is [shares_remaining_norm, time_remaining_norm, current_vol, vol_history(10)], its action is a fraction of remaining shares to sell this step, and its reward is the realized execution cost: -(η·v² + v·vol·noise), where v is shares sold. Any unsold shares at the final step trigger a large liquidation penalty to enforce full execution.

The observation contains 10 steps of vol history. The naive approach is to flatten it into a plain MLP input, which works but throws away the sequential structure. Instead, I wrote a custom SB3 `BaseFeaturesExtractor` that splits the observation in two: the scalar features pass through a linear layer and the vol history passes through a one-layer GRU, with the outputs concatenated into a single feature vector feeding PPO's actor and critic heads. The GRU hidden size of 32 and history length of 10 were reasonable defaults rather than tuned values. A single GRU layer was a deliberate choice, as a two-layer GRU would have more capacity
but would require significantly more training to converge, and 200,000 PPO steps is already pushing it.

The second RL agent is TQC (Truncated Quantile Critics) from `sb3-contrib`. Where standard actor-critic methods estimate state-action values as a single number, TQC models the return distribution as a set of quantiles and then drops the top ones before computing the value estimate. This deliberate pessimism about best-case outcomes reduces overestimation bias and tends to produce more stable training on continuous control tasks. TQC is built on SAC, which is off-policy — it stores experience in a replay buffer and reuses each transition many times, making it significantly more sample-efficient than PPO for short-episode tasks like this one. I used a standard MLP policy with default hyperparameters. An Optuna search over learning rate, batch size, and top_quantiles_to_drop is implemented in `agents/optimize_tqc.py` but not run at full scale — that search space is there for anyone with the compute budget. TQC was trained for 500,000 steps, which took two hours on an M4 MacBook Pro. At 200,000 steps the agent was worse than TWAP, which is humbling but not surprising for off-policy methods early in training.

Finally, all five strategies — TWAP, AC with constant vol, AC with rolling vol, PPO+GRU, and TQC — were evaluated over 500 episodes on the same environment to produce the benchmark comparison.

---

## Results

| Strategy         | Mean IS   | Std     | CVaR95    |
|-----------------|-----------|---------|-----------|
| TWAP            | 38,464.63 | 125.48  | 38,657.66 |
| AC constant vol | 38,450.37 | 124.21  | 38,659.74 |
| AC rolling vol  | 38,457.78 | 123.37  | 38,658.51 |
| PPO+GRU         | 37,818.72 | 115.39  | 38,009.16 |
| TQC             | 37,947.82 | 264.43  | 38,425.40 |

Note that Lower IS is better.

PPO+GRU wins on every metric: mean cost, variance, and tail risk. A 1.7% improvement over TWAP. TQC has a slightly higher mean IS than PPO but twice the variance, which is the kind of thing a risk desk notices.

The result I didn't expect is that AC rolling vol is almost indistinguishable from AC constant vol. The OU process mean-reverts quickly enough at 15-minute timescales that knowing the current vol barely helps the analytical model. The stochastic vol twist does make the environment harder, which forces the RL agents to learn something real but is nearly sidestepped by the AC model which just assumes the mean.

---

## Conclusions

The GRU matters! Feeding vol history as a sequence rather than a flat vector gave PPO a meaningful edge over both the analytical benchmarks and TQC's MLP. Whether that's the GRU encoding genuine temporal structure or just a better-parameterized policy is hard to say at 200k training steps. I suspect it's some of both.

PPO with a large reward scale is treacherous. The first training run produced an agent that learned to never sell anything, collecting the 1,000,000-unit forced liquidation penalty every episode with perfect consistency. Mean IS of exactly 1,000,000, standard deviation of 0.00. The fix was `VecNormalize` with reward clipping. This, apparently, is a known failure mode of on-policy methods and I'm documenting it here because it cost me a training run.

TQC took two hours to train 500,000 steps on a MacBook Pro (M4 chip). At 200,000 steps it was worse than TWAP. I know because I ran it and waited. If you're reproducing this, save the model after training.

---

## Shortcomings

The reward function has no variance penalty. Adding `-$\lambda$ · trajectory_variance` would push both agents toward consistent execution rather than just low average cost. I faced a similar shortfall in my Brownian-RL project (feel free to take a look at github.com/cjarmaly). I think it could close the CVaR gap between PPO and TQC significantly.

The Optuna study is implemented but not fully run. A proper search over GRU hidden size and history length for PPO, in parallel with the TQC hyperparameter search, would tell you whether the GRU result holds up or was lucky at the default settings.

The environment resets vol to the OU mean at the start of every episode. A more realistic version would initialize vol from a stationary OU draw, which would stress-test the agents' ability to handle high-vol regimes at episode start, which is in theory the scenario where execution decisions actually matter most.

---

## Project Structure

```
implementation-shortfall-rl/
├── data/
│   ├── calibration.py      # SPY data fetch, realized vol, OU calibration
│   └── constants.py        # calibrated OU params, execution params
├── models/
│   ├── almgren_chriss.py   # AC trajectory and expected IS
│   └── gru_policy.py       # custom GRU feature extractor + ActorCriticPolicy
├── envs/
│   └── execution_env.py    # Gymnasium execution environment
├── agents/
│   ├── tqc_agent.py        # TQC training
│   └── optimize_tqc.py     # Optuna hyperparameter search
├── tests/
│   ├── test_almgren_chriss.py
│   └── test_execution_env.py
└── experiments/
    ├── ac_trajectory/       # AC trajectory shapes
    └── benchmark/           # four-way comparison
```

## Setup

```bash
git clone https://github.com/your-username/implementation-shortfall-rl
cd implementation-shortfall-rl
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib yfinance torch stable-baselines3 sb3-contrib gymnasium optuna
```

Run the benchmark:

```bash
python -m experiments.benchmark.run
```

Run tests:

```bash
pytest tests/ -v
```