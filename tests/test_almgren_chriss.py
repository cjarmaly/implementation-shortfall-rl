import numpy as np
import pytest
from models.almgren_chriss import ac_optimal_trajectory, ac_expected_IS
from experiments.helpers import DEFAULT_EXEC_PARAMS, DEFAULT_OU_PARAMS

p = DEFAULT_EXEC_PARAMS
sigma = DEFAULT_OU_PARAMS["mu"]

def test_trajectory_starts_at_X():
    x = ac_optimal_trajectory(sigma, p["X"], p["T"], p["eta"], p["gamma"], p["lambda_"])
    assert np.isclose(x[0], p["X"])

def test_trajectory_ends_at_zero():
    x = ac_optimal_trajectory(sigma, p["X"], p["T"], p["eta"], p["gamma"], p["lambda_"])
    assert np.isclose(x[-1], 0, atol=1e-6)

def test_lambda_zero_recovers_twap():
    x = ac_optimal_trajectory(sigma, p["X"], p["T"], p["eta"], p["gamma"], lambda_=1e-10)
    steps = np.diff(x)
    assert np.allclose(steps, steps[0], rtol=1e-3)

def test_high_lambda_frontloads():
    x = ac_optimal_trajectory(sigma, p["X"], p["T"], p["eta"], p["gamma"], lambda_=1e-3)
    assert abs(x[0] - x[1]) > abs(x[-2] - x[-1])

def test_IS_positive():
    IS = ac_expected_IS(sigma, p["X"], p["T"], p["eta"], p["gamma"], p["lambda_"])
    assert IS > 0