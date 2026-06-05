"""因子预处理与线性化相关的 Polars 表达式（用于 ``with_columns`` 等）。"""

from __future__ import annotations

from collections.abc import Callable
from typing import overload

import polars as pl


def _is_finite(col: str) -> pl.Expr:
    return pl.col(col).is_not_null() & pl.col(col).is_finite()


def factor_group_expr(
    factor_col: str,
    n_groups: int,
    *,
    date_col: str = "trade_date",
) -> pl.Expr:
    """按因子从高到低等分分组，组 1 为因子最高。"""
    rank = pl.col(factor_col).rank(method="ordinal", descending=True).over(date_col)
    n = pl.len().over(date_col).cast(pl.Float64)
    bucket = ((rank - 1) * n_groups / n).floor().cast(pl.Int64)
    return pl.when(bucket >= n_groups).then(n_groups).otherwise(bucket + 1)


def factor_percentile_expr(
    factor_col: str,
    *,
    date_col: str = "trade_date",
) -> pl.Expr:
    """因子截面百分位（100 为最高）。"""
    rank = pl.col(factor_col).rank(method="average", descending=True).over(date_col)
    n = pl.len().over(date_col).cast(pl.Float64)
    return (n - rank + 0.5) / n * 100.0


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


@overload
def piecewise_linear_regression_expr(
    df: pl.DataFrame,
    factor_col: str,
    ret_col: str,
    *,
    output_col: str | None = None,
    date_col: str = "trade_date",
    stock_col: str = "stock_code",
    bench_ret_col: str | None = None,
    n_groups: int = 100,
    lookback_days: int = 120,
    decay: int = 2,
    trading_days: int = 252,
) -> pl.DataFrame: ...


@overload
def piecewise_linear_regression_expr(
    factor_col: str,
    ret_col: str,
    *,
    output_col: str | None = None,
    date_col: str = "trade_date",
    stock_col: str = "stock_code",
    bench_ret_col: str | None = None,
    n_groups: int = 100,
    lookback_days: int = 120,
    decay: int = 2,
    trading_days: int = 252,
) -> Callable[[pl.DataFrame], pl.DataFrame]: ...


def piecewise_linear_regression_expr(
    df_or_factor_col: pl.DataFrame | str,
    factor_col_or_ret_col: str,
    ret_col: str | None = None,
    *,
    output_col: str | None = None,
    date_col: str = "trade_date",
    stock_col: str = "stock_code",
    bench_ret_col: str | None = None,
    n_groups: int = 100,
    lookback_days: int = 120,
    decay: int = 2,
    trading_days: int = 252,
) -> pl.DataFrame | Callable[[pl.DataFrame], pl.DataFrame]:
    """分段线性回归，用于 ``df.pipe(...)`` 链式调用。

    算法需跨日历史面板估计映射，无法写成单行 ``pl.Expr``。支持两种调用方式：

    示例::

        from factorresearch.expr import (
            standardize_cross_section_expr,
            piecewise_linear_regression_expr,
        )

        # 推荐：Polars pipe 风格（df 由 pipe 传入）
        panel = (
            df.with_columns(
                standardize_cross_section_expr("factor").alias("factor_z"),
            )
            .pipe(
                piecewise_linear_regression_expr,
                "factor_z",
                "fut_ret",
                output_col="factor_plr",
            )
        )

        # 亦可先绑定参数再 pipe
        panel = df.pipe(
            piecewise_linear_regression_expr("factor_z", "fut_ret"),
        )
    """
    from research.linearization import piecewise_linear_regression

    kwargs = {
        "output_col": output_col,
        "date_col": date_col,
        "stock_col": stock_col,
        "bench_ret_col": bench_ret_col,
        "n_groups": n_groups,
        "lookback_days": lookback_days,
        "decay": decay,
        "trading_days": trading_days,
    }

    if isinstance(df_or_factor_col, pl.DataFrame):
        if ret_col is None:
            raise TypeError("ret_col is required")
        return piecewise_linear_regression(
            df_or_factor_col,
            factor_col=factor_col_or_ret_col,
            ret_col=ret_col,
            **kwargs,
        )

    factor_col = df_or_factor_col

    def _apply(df: pl.DataFrame) -> pl.DataFrame:
        return piecewise_linear_regression(
            df,
            factor_col=factor_col,
            ret_col=factor_col_or_ret_col,
            **kwargs,
        )

    return _apply
