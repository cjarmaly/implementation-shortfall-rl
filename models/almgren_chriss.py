import numpy as np

def ac_optimal_trajectory(sigma, X, T, eta, gamma, lambda_):
    """
    Compute the optimal trajectory for the Almgren-Chriss model
    theta: OU parameter
    mu: mean
    sigma: volatility
    X: total shares to execute
    T: number of 15-min bars in a trading day
    eta: temporary impact coefficient
    gamma: permanent impact coefficient
    lambda_: risk aversion
    returns: optimal trajectory
    """

    # compute kappa
    kappa = np.sqrt(lambda_ * sigma**2 / eta)

    #compute x(t) for each time step
    t = np.linspace(0, T, T + 1)
    x = X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)

    return x

def ac_expected_IS(sigma, X, T, eta, gamma, lambda_):
    """
    Returns expected implementation shortfall for the Almgren-Chriss model
    """
    kappa = np.sqrt(lambda_ * sigma**2 / eta)

    # compute IS using closed form
    IS = 0.5 * gamma * X**2 + (lambda_ * sigma**2 * X**2) / (2 * kappa) * (np.cosh(kappa * T) / np.sinh(kappa * T))
    
    return IS