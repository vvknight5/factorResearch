import math

import polars as pl

from plotting import plot_group_backtest
from research.utils import resolve_factor_name

_TRADING_DAYS = 252


def _validate_factor_df(
    df: pl.DataFrame,
    factor_col: str,
    ret_col: str,
    date_col: str,
) -> None:
    required = (factor_col, ret_col, date_col, "stock_code")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"df 缺少必需列: {', '.join(missing)}")


def _assign_factor_groups(
    df: pl.DataFrame,
    n: int,
    factor_col: str,
    ret_col: str,
    date_col: str,
) -> pl.DataFrame:
    labels = [str(i) for i in range(1, n + 1)]
    return (
        df.filter(pl.col(factor_col).is_not_null(), pl.col(ret_col).is_not_null())
        .with_columns(
            pl.col(factor_col)
            .qcut(n, labels=labels, allow_duplicates=True)
            .over(date_col)
            .alias("group"),
        )
        .with_columns(pl.col("group").cast(pl.Utf8).cast(pl.Int64))
        .filter(pl.col("group").is_not_null())
    )


def _daily_group_returns(
    grouped: pl.DataFrame,
    ret_col: str,
    date_col: str,
) -> pl.DataFrame:
    return (
        grouped.group_by(date_col, "group")
        .agg(pl.col(ret_col).mean().alias("daily_ret"))
        .sort(date_col, "group")
    )


def _with_cumulative_returns(daily: pl.DataFrame) -> pl.DataFrame:
    return daily.with_columns(
        pl.col("daily_ret").cum_sum().over("group").alias("cum_ret"),
    )


def _max_drawdown(daily_ret: pl.Series) -> float:
    cum = daily_ret.cum_sum()
    peak = cum.cum_max()
    return (cum - peak).min()


def _profit_loss_ratio(daily_ret: pl.Series) -> float:
    wins = daily_ret.filter(daily_ret > 0)
    losses = daily_ret.filter(daily_ret < 0)
    if wins.len() == 0 or losses.len() == 0:
        return float("nan")
    return wins.mean() / abs(losses.mean())


def _group_performance(
    daily: pl.DataFrame,
    date_col: str,
    *,
    trading_days: int = _TRADING_DAYS,
) -> pl.DataFrame:
    rows: list[dict[str, float | int]] = []
    groups = [g for g in daily["group"].unique().to_list() if g is not None]
    for g in sorted(groups):
        s = daily.filter(pl.col("group") == g).sort(date_col)
        rets = s["daily_ret"]
        n = rets.len()
        if n == 0:
            continue

        mean_daily = rets.mean()
        cum_return = rets.sum()
        std_daily = rets.std()
        annual_return = mean_daily * trading_days if n > 0 else float("nan")
        annual_vol = (
            std_daily * math.sqrt(trading_days)
            if std_daily is not None and not math.isnan(std_daily)
            else float("nan")
        )
        if annual_vol is None or annual_vol == 0 or math.isnan(annual_vol):
            sharpe = float("nan")
        else:
            sharpe = annual_return / annual_vol

        rows.append(
            {
                "group": g,
                "cum_return": cum_return,
                "annual_return": annual_return,
                "annual_vol": annual_vol,
                "sharpe": sharpe,
                "max_drawdown": _max_drawdown(rets),
                "mean_daily_return": mean_daily,
                "win_rate": rets.filter(rets > 0).len() / n,
                "profit_loss_ratio": _profit_loss_ratio(rets),
            }
        )
    return pl.DataFrame(rows).sort("group")


def factor_group_backtest(
    df: pl.DataFrame,
    n: int,
    *,
    factor_col: str = "factor",
    ret_col: str = "ret",
    date_col: str = "trade_date",
    factor_name: str | None = None,
    trading_days: int = _TRADING_DAYS,
    figsize: tuple[float, float] = (14, 5),
    show: bool = True,
) -> pl.DataFrame:
    """按截面因子分 N 组，计算组内等权日收益、累计收益曲线与回测指标。

    收益按单利计算：累计收益为日收益之和，年化为日均收益 × trading_days。

    参数 factor_col、ret_col 指定因子列与收益率列名；date_col 指定日期列（默认 trade_date）。
    factor_name 用于结果与图表中的因子标识（默认与 factor_col 相同）。
    另需包含 stock_code 列。返回每组回测表现（group 从 1 到 N，1 为因子最小组）。
    """
    if n < 1:
        raise ValueError("n 必须 >= 1")

    _validate_factor_df(df, factor_col, ret_col, date_col)

    grouped = _assign_factor_groups(df, n, factor_col, ret_col, date_col)
    daily = _daily_group_returns(grouped, ret_col, date_col)
    cum = _with_cumulative_returns(daily)
    perf = _group_performance(daily, date_col, trading_days=trading_days)
    name = resolve_factor_name(factor_name, factor_col)
    perf = perf.with_columns(pl.lit(name).alias("factor_name")).select(
        "factor_name",
        pl.all().exclude("factor_name"),
    )

    if show:
        plot_group_backtest(
            cum,
            perf,
            n,
            date_col=date_col,
            figsize=figsize,
            factor_name=name,
        )

    return perf
