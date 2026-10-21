"""
Mathematical and empirical constants from Zhang & Lee (2026).
"""

# Default parameters for Proposition 5.6 & Table 3
DEFAULT_R = 0.1             # Overthinking ratio
DEFAULT_GAMMA = 2.0         # Ambiguity scaling exponent
DEFAULT_LAMBDA_BETA = 0.01  # Normalized cost of thinking
DEFAULT_BETA = 1024         # Token scale parameter
DEFAULT_EPSILON = 0.35      # Give-up error tolerance threshold

# Nonparametric budget grid B in {0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}
BUDGET_GRID = [0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# Confidence binning for empirical budget tables
CONFIDENCE_BINS = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.00),
]

# Benchmark Task IDs
TASK_BOOLQ = "T1_BoolQ"
TASK_GSM8K = "T2_GSM8K"
TASK_CRT = "T3_CRT"
