# Experiment 02: Four-Way Benchmark — TWAP vs AC vs PPO+GRU vs TQC
 
## What We Did
 
Compared four execution strategies on the same stochastic-vol environment over 500 episodes each:
 
- **TWAP**: sell 1/T of remaining shares per step.
- **AC constant vol**: Almgren-Chriss optimal trajectory, recomputed each step using the OU long-run mean (σ = 0.0623) as a fixed vol estimate.
- **AC rolling vol**: same as above but feeds the live current vol from the observation into the trajectory computation.
- **PPO+GRU**: on-policy RL with a custom GRU feature extractor that reads the 10-step vol history as a sequence before feeding into the actor/critic heads.
- **TQC**: off-policy RL with truncated quantile critics (SAC variant), standard MLP policy.
## Results
 
| Strategy         | Mean IS   | Std     | CVaR95    |
|-----------------|-----------|---------|-----------|
| TWAP            | 38,464.63 | 125.48  | 38,657.66 |
| AC constant vol | 38,450.37 | 124.21  | 38,659.74 |
| AC rolling vol  | 38,457.78 | 123.37  | 38,658.51 |
| PPO+GRU         | 37,818.72 | 115.39  | 38,009.16 |
| TQC             | 37,947.82 | 264.43  | 38,425.40 |
 
Note that lower is better.
 
## What the Numbers Say
 
PPO+GRU wins on every metric. Lowest mean IS, lowest standard deviation, lowest CVaR95. A 1.7% improvement over TWAP sounds modest upon first glance, but at an instiutitonal scale, 1.7% is not nothing.
 
TQC has the second-best mean IS but twice the variance of PPO. The distributional critics help on average cost but introduce execution variability that is shown in the risk figure: CVaR95 of 38,425 vs PPO's 38,009 is a real gap.
 
The AC rolling vol result is nearly identical to AC constant vol (38,457 vs 38,450). The OU process mean-reverts fast enough at these timescales that knowing the current vol is barely better than assuming it's at the mean. The stochastic vol twist adds complexity to the environment, which forced the RL agents to actually learn something, but it appears that the analytical model nearly sidesteps it.
 
TWAP is last. As advertised.
 
## Getting Here Was Not Painless
 
The first PPO run collapsed immediately — the agent learned to never sell anything, collecting the 1,000,000-unit forced liquidation penalty every episode and achieving a mean IS of exactly 1,000,000 with zero standard deviation. The plot had PPO at 1,000,000 and everything else at 38,000, which produced an x-axis ranging to 1e6 and five invisible histograms huddled in the leftmost pixel.Impressive in the wrong direction. The fix was wrapping the environment in `VecNormalize` to scale rewards to unit variance during training, then evaluating on the raw environment.
 
TQC took two hours to train 500,000 steps at roughly 90 fps. The first run at 200,000 steps produced a mean IS of 59,000 (worse than TWAP) because the agent was still exploring. A second run at 500,000 steps converged. If you are running this on a laptop, plan accordingly.

 
## What I'd Do Next
 
Looking forward, I immediately think of three things:
 
The GRU is reading a 10-step vol history with a hidden size of 32. That's a reasonable default but notably untuned. The Optuna study in `agents/optimize_tqc.py` searches over TQC hyperparameters — a parallel search over GRU hidden size and history length for PPO would be a rigorous next step.
 
The AC rolling vol result suggests the environment's vol dynamics aren't volatile enough to stress the analytical model. Recalibrating with a higher `ou_sigma` or shorter mean-reversion time would force a wider spread between the constant-vol and rolling-vol AC strategies, and give the RL agents more signal to work with.
 
And the reward function is per-step realized cost: `-(eta*v^2 + v*vol*noise)`. There's no explicit penalty for variance across the trajectory. Adding `-lambda * trajectory_variance` to the reward would push PPO toward the consistency. The CVaR gap between PPO and TQC suggests this matters.