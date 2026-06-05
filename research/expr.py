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


def _as_dataframe(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    if isinstance(frame, pl.LazyFrame):
        return frame.collect()
    return frame


def _restore_frame_type(
    frame: pl.DataFrame | pl.LazyFrame,
    result: pl.DataFrame,
) -> pl.DataFrame | pl.LazyFrame:
    if isinstance(frame, pl.LazyFrame):
        return pl.LazyFrame(result)
    return result


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
    df: pl.LazyFrame,
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
) -> pl.LazyFrame: ...


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
) -> Callable[[pl.DataFrame | pl.LazyFrame], pl.DataFrame | pl.LazyFrame]: ...


def piecewise_linear_regression_expr(
    df_or_factor_col: pl.DataFrame | pl.LazyFrame | str,
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
) -> pl.DataFrame | pl.LazyFrame | Callable[[pl.DataFrame | pl.LazyFrame], pl.DataFrame | pl.LazyFrame]:
    """分段线性回归，用于 ``df.pipe(...)`` 链式调用。

    算法需跨日历史面板估计映射，无法写成单行 ``pl.Expr``。传入 ``LazyFrame`` 时会在
    内部 ``collect()`` 完成计算，但返回类型与输入一致，便于 lazy 管道继续链式调用。
    支持两种调用方式：

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

    if isinstance(df_or_factor_col, (pl.DataFrame, pl.LazyFrame)):
        if ret_col is None:
            raise TypeError("ret_col is required")
        result = piecewise_linear_regression(
            _as_dataframe(df_or_factor_col),
            factor_col=factor_col_or_ret_col,
            ret_col=ret_col,
            **kwargs,
        )
        return _restore_frame_type(df_or_factor_col, result)

    factor_col = df_or_factor_col

    def _apply(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        result = piecewise_linear_regression(
            _as_dataframe(df),
            factor_col=factor_col,
            ret_col=factor_col_or_ret_col,
            **kwargs,
        )
        return _restore_frame_type(df, result)

    return _apply


def _normalize_base_factors(base_factors: list[str] | str) -> list[str]:
    if isinstance(base_factors, str):
        return [base_factors]
    return list(base_factors)


def _orthogonalize_univariate_expr(
    new_factor: str,
    base_factor: str,
    *,
    date_col: str = "trade_date",
) -> pl.Expr:
    """单基准因子截面正交化；与 ``neutralize_market_cap_expr`` 算法相同。"""
    return neutralize_market_cap_expr(new_factor, base_factor, date_col=date_col)


@overload
def orthogonalize_factor_expr(
    new_factor: str,
    base_factor: str,
) -> pl.Expr: ...


@overload
def orthogonalize_factor_expr(
    df: pl.DataFrame,
    new_factor: str,
    base_factors: list[str] | str,
    ret_col: str,
    *,
    output_col: str | None = None,
    date_col: str = "trade_date",
    stock_col: str = "stock_code",
) -> pl.DataFrame: ...


@overload
def orthogonalize_factor_expr(
    df: pl.LazyFrame,
    new_factor: str,
    base_factors: list[str] | str,
    ret_col: str,
    *,
    output_col: str | None = None,
    date_col: str = "trade_date",
    stock_col: str = "stock_code",
) -> pl.LazyFrame: ...


@overload
def orthogonalize_factor_expr(
    new_factor: str,
    base_factors: list[str] | str,
    ret_col: str,
    *,
    output_col: str | None = None,
    date_col: str = "trade_date",
    stock_col: str = "stock_code",
) -> Callable[[pl.DataFrame | pl.LazyFrame], pl.DataFrame | pl.LazyFrame]: ...


def _resolve_orthogonalize_call(
    df_or_new_factor: str | pl.DataFrame | pl.LazyFrame,
    new_factor_or_base_factors: str | list[str],
    base_factors_or_ret_col: list[str] | str | None,
    ret_col: str | None,
) -> tuple[str, list[str], str] | None:
    """解析 pipe / 直接调用参数；返回 None 表示进入单基准 ``pl.Expr`` 模式。"""
    if isinstance(df_or_new_factor, (pl.DataFrame, pl.LazyFrame)):
        if not isinstance(new_factor_or_base_factors, str):
            raise TypeError("new_factor 必须为 str")
        if base_factors_or_ret_col is None:
            raise TypeError("base_factors is required")
        if ret_col is None:
            raise TypeError("ret_col is required")
        return (
            new_factor_or_base_factors,
            _normalize_base_factors(base_factors_or_ret_col),
            ret_col,
        )

    if not isinstance(df_or_new_factor, str):
        raise TypeError("new_factor 必须为 str")

    new_factor = df_or_new_factor

    if base_factors_or_ret_col is None and ret_col is None:
        factors = _normalize_base_factors(new_factor_or_base_factors)
        if len(factors) != 1:
            raise TypeError("多基准因子需通过 pipe 调用并传入 ret_col")
        return None

    if isinstance(new_factor_or_base_factors, list):
        rc = (
            base_factors_or_ret_col
            if isinstance(base_factors_or_ret_col, str)
            else ret_col
        )
        if rc is None:
            raise TypeError("ret_col is required")
        return new_factor, new_factor_or_base_factors, rc

    if not isinstance(new_factor_or_base_factors, str):
        raise TypeError("base_factors 必须为 str 或 list[str]")

    if isinstance(base_factors_or_ret_col, str) and ret_col is None:
        return new_factor, [new_factor_or_base_factors], base_factors_or_ret_col

    if ret_col is None:
        raise TypeError("ret_col is required")
    return new_factor, [new_factor_or_base_factors], ret_col


def orthogonalize_factor_expr(
    df_or_new_factor: str | pl.DataFrame | pl.LazyFrame,
    new_factor_or_base_factors: str | list[str],
    base_factors_or_ret_col: list[str] | str | None = None,
    ret_col: str | None = None,
    *,
    output_col: str | None = None,
    date_col: str = "trade_date",
    stock_col: str = "stock_code",
) -> (
    pl.Expr
    | pl.DataFrame
    | pl.LazyFrame
    | Callable[[pl.DataFrame | pl.LazyFrame], pl.DataFrame | pl.LazyFrame]
):
    """因子正交化表达式接口。

    - **单基准因子**：``orthogonalize_factor_expr("roe", "mom_20")`` 返回 ``pl.Expr``，
      可直接用于 ``with_columns``。
    - **多基准因子**：需跨列截面多元 OLS，使用 ``df.pipe(...)`` 链式调用。
      传入 ``LazyFrame`` 时内部 ``collect()``，返回类型与输入一致。

    示例::

        from factorresearch.expr import orthogonalize_factor_expr

        # 单基准：with_columns
        panel = df.with_columns(
            orthogonalize_factor_expr("roe", "mom_20").alias("roe_orth"),
        )

        # 多基准：pipe
        panel = df.pipe(
            orthogonalize_factor_expr,
            "roe",
            ["mom_20", "ep"],
            "fut_ret",
            output_col="roe_orth",
        )

        # 亦可先绑定参数再 pipe
        panel = df.pipe(
            orthogonalize_factor_expr("roe", ["mom_20", "ep"], "fut_ret"),
        )
    """
    from research.orthogonalization import orthogonalize_factor

    resolved = _resolve_orthogonalize_call(
        df_or_new_factor,
        new_factor_or_base_factors,
        base_factors_or_ret_col,
        ret_col,
    )

    if resolved is None:
        factors = _normalize_base_factors(new_factor_or_base_factors)
        return _orthogonalize_univariate_expr(
            df_or_new_factor,
            factors[0],
            date_col=date_col,
        )

    new_factor, base_factors, rc = resolved
    kwargs = {
        "output_col": output_col,
        "date_col": date_col,
        "stock_col": stock_col,
    }

    def _run(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        result = orthogonalize_factor(
            _as_dataframe(frame),
            base_factors=base_factors,
            new_factor=new_factor,
            ret_col=rc,
            **kwargs,
        )
        return _restore_frame_type(frame, result)

    if isinstance(df_or_new_factor, (pl.DataFrame, pl.LazyFrame)):
        return _run(df_or_new_factor)

    return _run
