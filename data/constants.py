# Calibrated OU parameters for SPY intraday vol (15-minute bars, 60 days)
OU_PARAMS = {
    "theta": 0.1139,
    "mu":    0.0623,
    "sigma": 0.0045,
}

# Data parameters
DATA_PARAMS = {
    "ticker":   "SPY",
    "period":   "60d",
    "interval": "15m",
    "window":   5,      # rolling vol window in bars
}

# Execution parameters (used in Modules 2+)
EXECUTION_PARAMS = {
    "X":      10000,     # total shares to execute
    "T":      26,        # number of 15-min bars in a trading day (6.5hrs)
    "eta":    0.01,      # temporary impact coefficient
    "gamma":  0.001,     # permanent impact coefficient
    "lambda_": 1e-6,     # risk aversion
}