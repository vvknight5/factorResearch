"""因子研究工具包（与 PyPI 分发名 factorresearch 对齐的导入路径）。"""

from research.evaluation import evaluate_factor, evaluate_factor_by_period, plot_factor_ic
from research.expr import neutralize_market_cap_expr, standardize_cross_section_expr
from research.group_backtest import factor_group_backtest
from research.preprocessing import neutralize_market_cap, standardize_cross_section

__all__ = [
    "evaluate_factor",
    "evaluate_factor_by_period",
    "plot_factor_ic",
    "factor_group_backtest",
    "neutralize_market_cap",
    "neutralize_market_cap_expr",
    "standardize_cross_section",
    "standardize_cross_section_expr",
]
