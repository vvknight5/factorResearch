import polars as pl

from research.expr import neutralize_market_cap_expr, standardize_cross_section_expr
from research.utils import resolve_factor_name


def _require_columns(df: pl.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"df 缺少必需列: {', '.join(missing)}")


def neutralize_market_cap(
    df: pl.DataFrame,
    *,
    factor_col: str,
    log_mv_col: str,
    date_col: str = "trade_date",
    stock_col: str = "stock_code",
    factor_name: str | None = None,
) -> pl.DataFrame:
    """截面市值中性化（DataFrame 封装）。

    等价于对 ``neutralize_market_cap_expr`` 的结果做 ``select``，并附加 ``factor_name`` 列。
    """
    _require_columns(df, (factor_col, log_mv_col, date_col, stock_col))
    name = resolve_factor_name(factor_name, factor_col)
    expr = neutralize_market_cap_expr(factor_col, log_mv_col, date_col=date_col)

    return df.select(
        pl.col(stock_col),
        pl.col(date_col),
        expr.alias(factor_col),
        pl.lit(name).alias("factor_name"),
    )


def standardize_cross_section(
    df: pl.DataFrame,
    *,
    factor_col: str,
    date_col: str = "trade_date",
    stock_col: str = "stock_code",
    factor_name: str | None = None,
) -> pl.DataFrame:
    """截面标准化（DataFrame 封装）。

    等价于对 ``standardize_cross_section_expr`` 的结果做 ``select``，并附加 ``factor_name`` 列。
    """
    _require_columns(df, (factor_col, date_col, stock_col))
    name = resolve_factor_name(factor_name, factor_col)
    expr = standardize_cross_section_expr(factor_col, date_col=date_col)

    return df.select(
        pl.col(stock_col),
        pl.col(date_col),
        expr.alias(factor_col),
        pl.lit(name).alias("factor_name"),
    )
