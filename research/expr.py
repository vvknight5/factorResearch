"""因子预处理相关的 Polars 表达式（用于 ``with_columns`` 等）。"""

import polars as pl


def _is_finite(col: str) -> pl.Expr:
    return pl.col(col).is_not_null() & pl.col(col).is_finite()


def neutralize_market_cap_expr(
    factor_col: str,
    log_mv_col: str,
    *,
    date_col: str = "trade_date",
) -> pl.Expr:
    """截面市值中性化表达式，可直接用于 ``with_columns``。

    在每个 ``date_col`` 分组内，用对数市值对因子做一元线性回归并返回残差。
    仅使用因子与对数市值均有限的样本估计截面统计量；无效行结果为 null。

    示例::

        df.with_columns(
            neutralize_market_cap_expr("factor", "log_mv").alias("factor_neu"),
        )
    """
    valid = _is_finite(factor_col) & _is_finite(log_mv_col)
    y = pl.when(valid).then(pl.col(factor_col))
    x = pl.when(valid).then(pl.col(log_mv_col))

    cov = pl.cov(y, x).over(date_col)
    var_x = x.var().over(date_col)
    beta = cov / var_x
    y_mean = y.mean().over(date_col)
    x_mean = x.mean().over(date_col)
    return pl.when(valid).then(
        (pl.col(factor_col) - y_mean) - beta * (pl.col(log_mv_col) - x_mean),
    )


def standardize_cross_section_expr(
    factor_col: str,
    *,
    date_col: str = "trade_date",
) -> pl.Expr:
    """截面 z-score 标准化表达式，可直接用于 ``with_columns``。

    在每个 ``date_col`` 分组内做 z-score；仅使用有限因子值估计均值与标准差；无效行结果为 null。

    示例::

        df.with_columns(
            standardize_cross_section_expr("factor").alias("factor_z"),
        )
    """
    valid = _is_finite(factor_col)
    y = pl.when(valid).then(pl.col(factor_col))
    mean = y.mean().over(date_col)
    std = y.std().over(date_col)
    return pl.when(valid).then((pl.col(factor_col) - mean) / std)
