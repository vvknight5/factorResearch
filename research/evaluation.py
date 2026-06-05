import math
from typing import Literal

import polars as pl

from research.expr import _is_finite
from research.utils import resolve_factor_name

Period = Literal["year", "month"]
CorrMethod = Literal["pearson", "spearman"]
_REQUIRED_COLS = ("factor", "ret", "trade_date", "stock_code")


def _daily_ic_df(
    df: pl.DataFrame,
    factor_col: str,
    ret_col: str,
    date_col: str,
    method: str,
) -> pl.DataFrame:
    return (
        df.filter(
            pl.col(factor_col).is_not_null(),
            pl.col(ret_col).is_not_null(),
        )
        .group_by(date_col)
        .agg(pl.corr(factor_col, ret_col, method=method).alias("ic"))
        .filter(pl.col("ic").is_finite())
    )


def _daily_ic(
    df: pl.DataFrame,
    factor_col: str,
    ret_col: str,
    date_col: str,
    method: str,
) -> pl.Series:
    return _daily_ic_df(df, factor_col, ret_col, date_col, method).get_column("ic")


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


def _period_expr(date_col: str, period: Period) -> pl.Expr:
    if period == "year":
        return pl.col(date_col).dt.year().alias("period")
    return pl.col(date_col).dt.strftime("%Y-%m").alias("period")


def _summarize_ic_period(group: pl.DataFrame) -> pl.DataFrame:
    ic_s = _ic_stats(group.get_column("ic"))
    rank_s = _ic_stats(group.get_column("rank_ic"))
    return pl.DataFrame(
        {
            "period": [group.get_column("period")[0]],
            "factor_name": [group.get_column("factor_name")[0]],
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


def _daily_ic_panel(
    df: pl.DataFrame,
    *,
    factor_col: str,
    ret_col: str,
    date_col: str,
) -> pl.DataFrame:
    ic_df = _daily_ic_df(df, factor_col, ret_col, date_col, method="pearson")
    rank_ic_df = _daily_ic_df(df, factor_col, ret_col, date_col, method="spearman").rename(
        {"ic": "rank_ic"}
    )
    return ic_df.join(rank_ic_df, on=date_col, how="inner").sort(date_col)


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


def evaluate_factor_by_period(
    df: pl.DataFrame,
    *,
    period: Period = "year",
    factor_col: str = "factor",
    ret_col: str = "ret",
    date_col: str = "trade_date",
    factor_name: str | None = None,
) -> pl.DataFrame:
    """按年或月汇总因子 IC / rank IC 统计量。

    period 为 ``year`` 时 period 列为年份（整数）；为 ``month`` 时为 ``YYYY-MM`` 字符串。
    其余参数与 evaluate_factor 相同。
    """
    if period not in ("year", "month"):
        raise ValueError("period 必须为 'year' 或 'month'")

    required = (factor_col, ret_col, date_col, "stock_code")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"df 缺少必需列: {', '.join(missing)}")

    name = resolve_factor_name(factor_name, factor_col)
    daily = _daily_ic_panel(
        df,
        factor_col=factor_col,
        ret_col=ret_col,
        date_col=date_col,
    )
    if daily.is_empty():
        return pl.DataFrame(
            schema={
                "period": pl.Int64 if period == "year" else pl.Utf8,
                "factor_name": pl.Utf8,
                "ic_mean": pl.Float64,
                "ic_std": pl.Float64,
                "ic_max": pl.Float64,
                "ic_min": pl.Float64,
                "ic_t_stat": pl.Float64,
                "icir": pl.Float64,
                "rank_ic_mean": pl.Float64,
                "rank_ic_t_stat": pl.Float64,
            }
        )

    daily = daily.with_columns(
        _period_expr(date_col, period),
        pl.lit(name).alias("factor_name"),
    )
    return (
        daily.group_by("period", maintain_order=True)
        .map_groups(_summarize_ic_period)
        .sort("period")
    )


def plot_factor_ic(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    ret_col: str = "ret",
    date_col: str = "trade_date",
    factor_name: str | None = None,
    figsize: tuple[float, float] = (14, 6.5),
    show: bool = True,
) -> pl.DataFrame:
    """计算日度 IC 并绘制 IC 图，返回含 cum_ic 的日度序列。"""
    from plotting import plot_ic

    required = (factor_col, ret_col, date_col, "stock_code")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"df 缺少必需列: {', '.join(missing)}")

    daily = _daily_ic_df(
        df, factor_col, ret_col, date_col, method="pearson"
    ).sort(date_col)
    if daily.is_empty():
        raise ValueError("无有效 IC 数据，无法绘图")

    name = resolve_factor_name(factor_name, factor_col)
    plot_ic(
        daily,
        date_col=date_col,
        factor_name=name,
        figsize=figsize,
        show=show,
    )
    return daily.with_columns(pl.col("ic").cum_sum().alias("cum_ic"))


def _daily_factor_correlation(
    df: pl.DataFrame,
    factor_a: str,
    factor_b: str,
    date_col: str,
    method: CorrMethod,
) -> pl.Series:
    valid = _is_finite(factor_a) & _is_finite(factor_b)
    return (
        df.filter(valid)
        .group_by(date_col)
        .agg(pl.corr(factor_a, factor_b, method=method).alias("corr"))
        .filter(pl.col("corr").is_finite())
        .get_column("corr")
    )


def factor_correlation_matrix(
    df: pl.DataFrame,
    factor_cols: list[str],
    *,
    date_col: str = "trade_date",
    method: CorrMethod = "pearson",
) -> pl.DataFrame:
    """计算因子间平均截面相关矩阵。

    对每个交易日分别计算因子对的截面相关，再对时间取平均。
    对角线为 1；矩阵对称。无效样本（null / NaN / Inf）不参与当日估计。

    参数
    ----
    factor_cols:
        因子列名列表，至少 1 列。
    date_col:
        日期列名（默认 ``trade_date``）。
    method:
        ``pearson`` 或 ``spearman``。
    """
    if not factor_cols:
        raise ValueError("factor_cols 至少包含 1 列")
    if method not in ("pearson", "spearman"):
        raise ValueError("method 必须为 'pearson' 或 'spearman'")

    missing = [c for c in (date_col, *factor_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"df 缺少必需列: {', '.join(missing)}")

    n = len(factor_cols)
    values: list[list[float | None]] = [[None] * n for _ in range(n)]

    for i, col_i in enumerate(factor_cols):
        values[i][i] = 1.0
        for j in range(i + 1, n):
            col_j = factor_cols[j]
            corr = _daily_factor_correlation(
                df, col_i, col_j, date_col, method
            )
            if corr.len() == 0:
                mean_corr = float("nan")
            else:
                mean_corr = corr.mean()
                if mean_corr is None or (
                    isinstance(mean_corr, float) and math.isnan(mean_corr)
                ):
                    mean_corr = float("nan")
            values[i][j] = mean_corr
            values[j][i] = mean_corr

    data: dict[str, pl.Series] = {
        "factor_name": pl.Series(factor_cols, dtype=pl.Utf8),
    }
    for j, col_j in enumerate(factor_cols):
        data[col_j] = pl.Series([row[j] for row in values], dtype=pl.Float64)

    return pl.DataFrame(data)
