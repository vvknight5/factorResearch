"""因子研究工具包（与 PyPI 分发名 factorresearch 对齐的导入路径）。"""

from research.evaluation import (
    evaluate_factor,
    evaluate_factor_by_period,
    factor_correlation_matrix,
    plot_factor_ic,
)
from research.expr import (
    factor_group_expr,
    factor_percentile_expr,
    neutralize_market_cap_expr,
    orthogonalize_factor_expr,
    piecewise_linear_regression_expr,
    standardize_cross_section_expr,
)
from research.group_backtest import factor_group_backtest
from research.linearization import piecewise_linear_regression
from research.orthogonalization import orthogonalize_factor
from research.preprocessing import neutralize_market_cap, standardize_cross_section
from plotting import plot_factor_correlation_matrix

__all__ = [
    "evaluate_factor",
    "evaluate_factor_by_period",
    "factor_correlation_matrix",
    "plot_factor_correlation_matrix",
    "plot_factor_ic",
    "factor_group_backtest",
    "factor_group_expr",
    "factor_percentile_expr",
    "neutralize_market_cap",
    "neutralize_market_cap_expr",
    "orthogonalize_factor_expr",
    "piecewise_linear_regression_expr",
    "standardize_cross_section",
    "standardize_cross_section_expr",
    "piecewise_linear_regression",
    "orthogonalize_factor",
]
