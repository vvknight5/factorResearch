"""非线性因子的线性化变换（如分段线性回归）。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

from research.expr import _is_finite, factor_group_expr, factor_percentile_expr


def _require_columns(df: pl.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"df 缺少必需列: {', '.join(missing)}")


def _ols(x: list[float], y: list[float]) -> tuple[float, float]:
    n = len(x)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return y[0], 0.0

    x_mean = sum(x) / n
    y_mean = sum(y) / n
    var_x = sum((xi - x_mean) ** 2 for xi in x)
    if var_x == 0:
        return y_mean, 0.0

    cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    beta = cov / var_x
    alpha = y_mean - beta * x_mean
    return alpha, beta


@dataclass(frozen=True)
class _Segment:
    intercept: float
    slope: float
    x_low: float
    x_high: float

    def predict(self, x: float) -> float:
        return self.intercept + self.slope * x


def _fit_piecewise_mapping(
    xs: list[float],
    ys: list[float],
) -> list[_Segment]:
    if len(xs) != len(ys) or len(xs) < 2:
        if len(xs) == 1:
            val = ys[0]
            return [_Segment(val, 0.0, xs[0], xs[0])]
        return []

    max_idx = max(range(len(ys)), key=lambda i: ys[i])
    min_idx = min(range(len(ys)), key=lambda i: ys[i])
    x_max = xs[max_idx]
    x_min = xs[min_idx]

    if x_max < x_min:
        x_max, x_min = x_min, x_max

    high: list[tuple[float, float]] = []
    mid: list[tuple[float, float]] = []
    low: list[tuple[float, float]] = []

    for x, y in zip(xs, ys):
        if x >= x_max:
            high.append((x, y))
        elif x <= x_min:
            low.append((x, y))
        else:
            mid.append((x, y))

    segments: list[_Segment] = []
    for bucket in (high, mid, low):
        if not bucket:
            continue
        bx = [p[0] for p in bucket]
        by = [p[1] for p in bucket]
        alpha, beta = _ols(bx, by)
        segments.append(_Segment(alpha, beta, min(bx), max(bx)))

    segments.sort(key=lambda s: s.x_low)
    return segments


def _predict_from_segments(x: float, segments: list[_Segment]) -> float | None:
    if not segments:
        return None
    if len(segments) == 1:
        return segments[0].predict(x)

    if x <= segments[0].x_high:
        return segments[0].predict(x)
    if x >= segments[-1].x_low:
        return segments[-1].predict(x)

    for seg in segments:
        if seg.x_low <= x <= seg.x_high:
            return seg.predict(x)

    for i in range(len(segments) - 1):
        left, right = segments[i], segments[i + 1]
        if left.x_high < x < right.x_low:
            x0, y0 = left.x_high, left.predict(left.x_high)
            x1, y1 = right.x_low, right.predict(right.x_low)
            if x1 == x0:
                return y0
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)

    return segments[-1].predict(x)


def _predict_many(pcts: list[float], segments: list[_Segment]) -> list[float | None]:
    return [_predict_from_segments(p, segments) for p in pcts]


def _estimate_mapping_from_precomputed(
    *,
    est_date,
    window_dates: list,
    group_x_by_date: dict[object, dict[int, float]],
    group_y_by_date: dict[object, dict[int, float]],
    n_groups: int,
    trading_days: int,
) -> list[_Segment] | None:
    xs_map = group_x_by_date.get(est_date)
    if not xs_map:
        return None

    y_sum: dict[int, float] = {g: 0.0 for g in range(1, n_groups + 1)}
    y_cnt: dict[int, int] = {g: 0 for g in range(1, n_groups + 1)}

    for window_date in window_dates:
        day_y = group_y_by_date.get(window_date)
        if not day_y:
            continue
        for g, val in day_y.items():
            y_sum[g] += val
            y_cnt[g] += 1

    xs: list[float] = []
    ys: list[float] = []
    for g in range(1, n_groups + 1):
        if g not in xs_map or y_cnt[g] == 0:
            continue
        xs.append(xs_map[g])
        ys.append(y_sum[g] / y_cnt[g] * trading_days)

    if len(xs) < 2:
        return None

    pairs = sorted(zip(xs, ys), key=lambda p: p[0], reverse=True)
    return _fit_piecewise_mapping([p[0] for p in pairs], [p[1] for p in pairs])


def _build_group_lookup(
    table: pl.DataFrame,
    *,
    date_col: str,
    group_col: str,
    value_col: str,
) -> dict[object, dict[int, float]]:
    lookup: dict[object, dict[int, float]] = {}
    for row in table.iter_rows(named=True):
        d = row[date_col]
        g = int(row[group_col])
        lookup.setdefault(d, {})[g] = float(row[value_col])
    return lookup


def piecewise_linear_regression(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    ret_col: str = "ret",
    output_col: str | None = None,
    date_col: str = "trade_date",
    stock_col: str = "stock_code",
    bench_ret_col: str | None = None,
    n_groups: int = 100,
    lookback_days: int = 120,
    decay: int = 2,
    trading_days: int = 252,
) -> pl.DataFrame:
    """分段线性回归线性化：在估计窗口内拟合因子分位与超额收益映射，输出转换因子。

    对交易日 ``t``，映射仅使用 ``[t - decay - lookback_days + 1, t - decay]`` 的数据估计，
    并应用于 ``t`` 日截面因子。返回输入 ``df`` 并追加转换后的因子列。

    参数
    ----
    factor_col:
        待转换因子列。
    ret_col:
        与 ``factor_col`` 对齐的收益率列。
    output_col:
        输出列名，默认 ``{factor_col}_plr``；若与 ``factor_col`` 同名则替换原因子列。
        尚无足够回溯窗口的行结果为 null。
    bench_ret_col:
        基准收益列；未指定时对 ``ret_col`` 按交易日等权平均作为基准。
    n_groups:
        估计映射时的截面分组数（研报默认 100）。
    lookback_days:
        估计映射时各组超额收益的回溯交易日数（研报默认 120）。
    decay:
        映射估计的最晚滞后交易日数（默认 2，即最晚使用 ``t - decay`` 的数据）。
    trading_days:
        年化换算使用的交易日数。
    """
    _require_columns(df, (factor_col, ret_col, date_col, stock_col))
    if n_groups < 3:
        raise ValueError("n_groups 至少为 3")
    if lookback_days < 1:
        raise ValueError("lookback_days 至少为 1")
    if decay < 0:
        raise ValueError("decay 不能为负数")

    out_col = output_col or f"{factor_col}_plr"

    valid = _is_finite(factor_col) & _is_finite(ret_col)
    panel = df.with_columns(
        pl.when(valid).then(pl.col(ret_col)).alias("_ret"),
        pl.when(_is_finite(factor_col)).then(pl.col(factor_col)).alias("_factor"),
    )

    if bench_ret_col is not None:
        _require_columns(df, (bench_ret_col,))
        panel = panel.with_columns(
            pl.when(_is_finite(bench_ret_col))
            .then(pl.col(bench_ret_col))
            .alias("_bench"),
        )
    else:
        panel = panel.with_columns(
            pl.when(valid)
            .then(pl.col("_ret").mean().over(date_col))
            .alias("_bench"),
        )

    panel = panel.with_columns(
        pl.when(valid & pl.col("_bench").is_not_null())
        .then(pl.col("_ret") - pl.col("_bench"))
        .alias("_excess"),
    )

    ready = panel.filter(
        _is_finite("_factor"),
        _is_finite("_excess"),
    ).with_columns(
        factor_group_expr("_factor", n_groups, date_col=date_col).alias("_group"),
        factor_percentile_expr("_factor", date_col=date_col).alias("_pct"),
    )

    if ready.is_empty():
        return _attach_output_column(
            df, None, out_col=out_col, factor_col=factor_col,
            date_col=date_col, stock_col=stock_col,
        )

    daily_group_excess = (
        ready.group_by(date_col, "_group")
        .agg(
            pl.col("_excess").mean().alias("_group_excess"),
            pl.len().alias("_group_size"),
        )
        .filter(pl.col("_group_size") >= 1)
    )

    est_group_x = (
        ready.group_by(date_col, "_group")
        .agg(pl.col("_pct").median().alias("_group_x"))
        .filter(pl.col("_group").is_not_null())
    )

    group_y_by_date = _build_group_lookup(
        daily_group_excess,
        date_col=date_col,
        group_col="_group",
        value_col="_group_excess",
    )
    group_x_by_date = _build_group_lookup(
        est_group_x,
        date_col=date_col,
        group_col="_group",
        value_col="_group_x",
    )

    pct_by_date: dict[object, tuple[list, list]] = {}
    for date_val, sub in ready.group_by(date_col, maintain_order=True):
        d = date_val[0]
        pct_by_date[d] = (
            sub.get_column(stock_col).to_list(),
            sub.get_column("_pct").to_list(),
        )

    dates = sorted(pct_by_date)
    transform_parts: list[pl.DataFrame] = []

    min_idx = decay + lookback_days - 1
    for t_idx in range(min_idx, len(dates)):
        t_date = dates[t_idx]
        est_idx = t_idx - decay
        est_date = dates[est_idx]
        window_dates = dates[est_idx - lookback_days + 1 : est_idx + 1]

        segments = _estimate_mapping_from_precomputed(
            est_date=est_date,
            window_dates=window_dates,
            group_x_by_date=group_x_by_date,
            group_y_by_date=group_y_by_date,
            n_groups=n_groups,
            trading_days=trading_days,
        )
        if segments is None:
            continue

        stocks, pcts = pct_by_date[t_date]
        rows = [
            (s, v)
            for s, v in zip(stocks, _predict_many(pcts, segments))
            if v is not None and math.isfinite(v)
        ]
        if not rows:
            continue
        out_stocks, out_values = zip(*rows)
        transform_parts.append(
            pl.DataFrame(
                {
                    date_col: [t_date] * len(out_stocks),
                    stock_col: list(out_stocks),
                    out_col: list(out_values),
                },
                schema={
                    date_col: panel.schema[date_col],
                    stock_col: panel.schema[stock_col],
                    out_col: pl.Float64,
                },
            )
        )

    if transform_parts:
        transformed_df = pl.concat(transform_parts)
        return _attach_output_column(
            df, transformed_df, out_col=out_col, factor_col=factor_col,
            date_col=date_col, stock_col=stock_col,
        )

    return _attach_output_column(
        df, None, out_col=out_col, factor_col=factor_col,
        date_col=date_col, stock_col=stock_col,
    )


def _attach_output_column(
    df: pl.DataFrame,
    transformed_df: pl.DataFrame | None,
    *,
    out_col: str,
    factor_col: str,
    date_col: str,
    stock_col: str,
) -> pl.DataFrame:
    if transformed_df is None:
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias(out_col))

    if out_col == factor_col:
        tmp_col = "_plr_tmp"
        cols = df.columns
        return (
            df.drop(factor_col)
            .join(
                transformed_df.rename({out_col: tmp_col}),
                on=[date_col, stock_col],
                how="left",
            )
            .rename({tmp_col: factor_col})
            .select(cols)
        )

    return df.join(transformed_df, on=[date_col, stock_col], how="left")
