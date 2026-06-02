import math

import polars as pl

from research.utils import resolve_factor_name

_REQUIRED_COLS = ("factor", "ret", "trade_date", "stock_code")


def _daily_ic(
    df: pl.DataFrame,
    factor_col: str,
    ret_col: str,
    date_col: str,
    method: str,
) -> pl.Series:
    return (
        df.filter(
            pl.col(factor_col).is_not_null(),
            pl.col(ret_col).is_not_null(),
        )
        .group_by(date_col)
        .agg(pl.corr(factor_col, ret_col, method=method).alias("ic"))
        .filter(pl.col("ic").is_finite())
        .get_column("ic")
    )


def _ic_stats(ic: pl.Series) -> dict[str, float]:
    ic = ic.filter(ic.is_finite())
    n = ic.len()
    if n == 0:
        nan = float("nan")
        return {
            "mean": nan,
            "std": nan,
            "max": nan,
            "min": nan,
            "t_stat": nan,
            "ir": nan,
        }

    mean = ic.mean()
    std = ic.std()
    ic_max = ic.max()
    ic_min = ic.min()

    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        nan = float("nan")
        return {
            "mean": nan,
            "std": nan,
            "max": ic_max,
            "min": ic_min,
            "t_stat": nan,
            "ir": nan,
        }

    if std is None or std == 0 or math.isnan(std):
        t_stat = float("nan")
        ir = float("nan")
    else:
        t_stat = mean / (std / math.sqrt(n))
        ir = mean / std

    return {
        "mean": mean,
        "std": std,
        "max": ic_max,
        "min": ic_min,
        "t_stat": t_stat,
        "ir": ir,
    }


def evaluate_factor(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    ret_col: str = "ret",
    date_col: str = "trade_date",
    factor_name: str | None = None,
) -> pl.DataFrame:
    """评价单因子：按交易日计算 IC / rank IC 序列，并汇总统计量。

    参数 factor_col、ret_col 指定因子列与收益率列名；date_col 指定日期列（默认 trade_date）。
    factor_name 用于结果中的因子标识（默认与 factor_col 相同）。
    另需包含 stock_code 列。
    """
    required = (factor_col, ret_col, date_col, "stock_code")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"df 缺少必需列: {', '.join(missing)}")

    ic = _daily_ic(df, factor_col, ret_col, date_col, method="pearson")
    rank_ic = _daily_ic(df, factor_col, ret_col, date_col, method="spearman")

    ic_s = _ic_stats(ic)
    rank_s = _ic_stats(rank_ic)
    name = resolve_factor_name(factor_name, factor_col)

    return pl.DataFrame(
        {
            "factor_name": [name],
            "ic_mean": [ic_s["mean"]],
            "ic_std": [ic_s["std"]],
            "ic_max": [ic_s["max"]],
            "ic_min": [ic_s["min"]],
            "ic_t_stat": [ic_s["t_stat"]],
            "icir": [ic_s["ir"]],
            "rank_ic_mean": [rank_s["mean"]],
            "rank_ic_t_stat": [rank_s["t_stat"]],
        }
    )
