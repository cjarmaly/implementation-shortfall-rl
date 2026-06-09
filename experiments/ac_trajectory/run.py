"""Experiment 01: AC optimal trajectory sanity check."""

import numpy as np
import matplotlib.pyplot as plt
from models.almgren_chriss import ac_optimal_trajectory
from experiments.helpers import DEFAULT_EXEC_PARAMS, DEFAULT_OU_PARAMS

def main():
    p = DEFAULT_EXEC_PARAMS
    sigma = DEFAULT_OU_PARAMS["mu"]

    x_passive   = ac_optimal_trajectory(sigma, p["X"], p["T"], p["eta"], p["gamma"], lambda_=1e-10)
    x_baseline  = ac_optimal_trajectory(sigma, p["X"], p["T"], p["eta"], p["gamma"], p["lambda_"])
    x_aggressive = ac_optimal_trajectory(sigma, p["X"], p["T"], p["eta"], p["gamma"], lambda_=1)
    x_very_aggressive = ac_optimal_trajectory(sigma, p["X"], p["T"], p["eta"], p["gamma"], lambda_=10)


    plt.figure(figsize=(10, 5))
    plt.plot(x_passive, label="TWAP (lambda~0)")
    plt.plot(x_baseline, label=f"AC baseline (lambda={p['lambda_']})")
    plt.plot(x_aggressive, label="Aggressive (lambda=1e-3)")
    plt.plot(x_very_aggressive, label="Very aggressive (lambda=10)")

    plt.title("AC Optimal Trajectories — Effect of Risk Aversion")
    plt.xlabel("Time step (15-min bars)")
    plt.ylabel("Remaining inventory")
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiments/ac_trajectory/ac_trajectories.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()