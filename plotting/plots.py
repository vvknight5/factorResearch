import polars as pl

_IC_AXIS_TICK_FONTSIZE = 16
_IC_AXIS_LABEL_FONTSIZE = 17
_IC_TITLE_FONTSIZE = 16
_IC_LEGEND_FONTSIZE = 15
_IC_FIGSIZE = (14, 6.5)
_IC_BAR_COLOR = "C0"
_IC_CUM_COLOR = "C3"
_LABEL_FONTSIZE = 14
_TICK_FONTSIZE = 12
_TITLE_FONTSIZE = 14
_LEGEND_FONTSIZE = 12


def _style_axis(
    ax,
    *,
    legend: bool = False,
) -> None:
    ax.tick_params(axis="both", labelsize=_TICK_FONTSIZE)
    ax.xaxis.label.set_fontsize(_LABEL_FONTSIZE)
    ax.yaxis.label.set_fontsize(_LABEL_FONTSIZE)
    ax.title.set_fontsize(_TITLE_FONTSIZE)
    if legend:
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            fontsize=_LEGEND_FONTSIZE,
        )


def hist_plot(
    df: pl.DataFrame,
    col: str,
    bins: int = 50,
    alpha: float = 0.7,
) -> None:
    """绘制列值的直方分布图。"""
    import matplotlib.pyplot as plt

    s = df[col].drop_nulls().to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(s, bins=bins, edgecolor="black", alpha=alpha)
    ax.set_xlabel(col)
    ax.set_ylabel("count")
    ax.set_title(f"{col} distribution")
    _style_axis(ax)
    fig.tight_layout()
    plt.show()


def plot_quantile_bucket_curve(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    N: int,
    *,
    figsize: tuple[float, float] = (8, 4),
    show: bool = True,
    title_msg: str = "",
) -> pl.DataFrame:
    """按 x_col 全样本分位切 N 组，组内 x 均值 vs y 均值（y 为 0/1 时即占比）。"""
    import matplotlib.pyplot as plt

    g = (
        df.filter(pl.col(x_col).is_not_null())
        .with_columns(
            pl.when(pl.col(x_col).count() > 1)
            .then(
                (pl.col(x_col).rank(method="average") - 1)
                / (pl.col(x_col).count() - 1)
            )
            .otherwise(0.0)
            .alias("x_pct"),
        )
        .with_columns(
            pl.col("x_pct").qcut(N, allow_duplicates=True).alias("pct_bin"),
        )
        .group_by("pct_bin")
        .agg(
            pl.col(x_col).mean().alias("x_mean"),
            pl.col(y_col).mean().alias("y_rate"),
        )
        .sort("x_mean")
    )
    if show:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(g["x_mean"], g["y_rate"], marker="o")
        ax.set_xlabel(f"{x_col} mean")
        ax.set_ylabel(f"{y_col} mean")
        ax.set_title(f"N = {N}, count = {df.height}" + title_msg)
        ax.grid(True, alpha=0.3)
        _style_axis(ax)
        fig.tight_layout()
        plt.show()
    return g


def plot_group_backtest(
    cum: pl.DataFrame,
    perf: pl.DataFrame,
    n: int,
    *,
    date_col: str = "trade_date",
    figsize: tuple[float, float] = (14, 5),
    factor_name: str | None = None,
) -> None:
    """绘制分组回测双子图：各组累计收益曲线、最终累计收益柱形图。"""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for g in sorted(cum["group"].unique().to_list()):
        s = cum.filter(pl.col("group") == g)
        ax1.plot(s[date_col], s["cum_ret"], label=f"G{g}")

    prefix = f"{factor_name} — " if factor_name else ""
    ax1.set_xlabel(date_col)
    ax1.set_ylabel("cumulative return")
    ax1.set_title(f"{prefix}group cumulative returns (N={n})")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)
    _style_axis(ax1, legend=True)

    groups = perf["group"].to_list()
    final_cum = perf["cum_return"].to_list()
    ax2.bar([str(g) for g in groups], final_cum, edgecolor="black")
    ax2.set_xlabel("group")
    ax2.set_ylabel("cumulative return")
    ax2.set_title(f"{prefix}final cumulative return by group")
    ax2.grid(True, alpha=0.3, axis="y")
    _style_axis(ax2)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.45)
    plt.show()


def plot_ic(
    daily: pl.DataFrame,
    *,
    date_col: str = "trade_date",
    ic_col: str = "ic",
    factor_name: str | None = None,
    figsize: tuple[float, float] = _IC_FIGSIZE,
    show: bool = True,
) -> None:
    """绘制日度 IC 图：左轴柱形图 + 均值虚线，右轴累计 IC 折线 + 零轴虚线。"""
    import math

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np

    missing = [c for c in (date_col, ic_col) if c not in daily.columns]
    if missing:
        raise ValueError(f"daily 缺少必需列: {', '.join(missing)}")
    if daily.is_empty():
        raise ValueError("daily 为空，无法绘图")

    daily = daily.sort(date_col)
    dates = daily[date_col].to_list()
    ic = daily[ic_col].to_numpy()
    cum_ic = np.cumsum(ic)
    mean_ic = float(np.mean(ic))
    n = ic.size
    std = float(np.std(ic, ddof=1)) if n > 1 else float("nan")
    if std == 0 or math.isnan(std):
        ic_t_stat = float("nan")
    else:
        ic_t_stat = mean_ic / (std / math.sqrt(n))

    fig, ax1 = plt.subplots(figsize=figsize)
    x = mdates.date2num(dates)
    if len(x) > 1:
        bar_width = min(0.8, 0.8 * float(np.min(np.diff(x))))
    else:
        bar_width = 0.8

    ax1.bar(x, ic, width=bar_width, color=_IC_BAR_COLOR, edgecolor="none")
    mean_line = ax1.axhline(
        mean_ic,
        color=_IC_BAR_COLOR,
        linestyle="--",
        linewidth=2.5,
    )
    ax1.legend(
        handles=[mean_line],
        labels=[f"mean IC = {mean_ic:.4f}"],
        loc="upper left",
        fontsize=_IC_LEGEND_FONTSIZE,
    )
    ax1.set_ylabel("daily IC", color=_IC_BAR_COLOR, fontsize=_IC_AXIS_LABEL_FONTSIZE)
    ax1.tick_params(
        axis="y",
        colors=_IC_BAR_COLOR,
        labelcolor=_IC_BAR_COLOR,
        labelsize=_IC_AXIS_TICK_FONTSIZE,
    )

    ax2 = ax1.twinx()
    ax2.plot(x, cum_ic, color=_IC_CUM_COLOR, linestyle="-")
    ax2.axhline(0, color=_IC_CUM_COLOR, linestyle="--")
    ax2.set_ylabel("cumulative IC", color=_IC_CUM_COLOR, fontsize=_IC_AXIS_LABEL_FONTSIZE)
    ax2.tick_params(
        axis="y",
        colors=_IC_CUM_COLOR,
        labelcolor=_IC_CUM_COLOR,
        labelsize=_IC_AXIS_TICK_FONTSIZE,
    )

    half_width = bar_width / 2
    ax1.set_xlim(x[0] - half_width, x[-1] + half_width)
    ax1.margins(x=0)

    ax1.set_xlabel(date_col, fontsize=_IC_AXIS_LABEL_FONTSIZE)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.tick_params(axis="x", rotation=45, labelsize=_IC_AXIS_TICK_FONTSIZE)

    prefix = f"{factor_name} — " if factor_name else ""
    ax1.set_title(
        f"{prefix}daily IC (IC t-stat = {ic_t_stat:.4f})",
        fontsize=_IC_TITLE_FONTSIZE,
    )

    fig.tight_layout()
    if show:
        plt.show()


def _correlation_heatmap_layout(
    n: int,
    max_label_len: int,
) -> dict[str, float | tuple[float, float]]:
    """按因子数量与标签长度估算热图布局参数。"""
    cell = 0.95
    label_pad = max_label_len * 0.08
    side = max(5.0, cell * n + 2.0 + label_pad)
    figsize = (side + 1.0, side)
    tick_fontsize = max(7.0, min(12.0, 13.5 - 0.45 * n))
    annotate_fontsize = max(7.0, min(12.0, 12.5 - 0.45 * n))
    left = min(0.42, 0.10 + max_label_len * 0.014)
    bottom = min(0.32, 0.12 + max_label_len * 0.010)
    return {
        "figsize": figsize,
        "tick_fontsize": tick_fontsize,
        "annotate_fontsize": annotate_fontsize,
        "left": left,
        "bottom": bottom,
        "right": 0.88,
    }


def plot_factor_correlation_matrix(
    corr: pl.DataFrame,
    *,
    factor_name_col: str = "factor_name",
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    annotate: bool | None = None,
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    show: bool = True,
) -> None:
    """绘制因子相关矩阵热图。

    输入通常为 ``factor_correlation_matrix`` 的返回值：含 ``factor_name`` 行标签列
    及各因子列构成的方阵。未指定 ``figsize`` 时，会根据因子数量与标签长度自动调整
    图像尺寸、字号与边距。
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if factor_name_col not in corr.columns:
        raise ValueError(f"corr 缺少必需列: {factor_name_col}")
    if corr.is_empty():
        raise ValueError("corr 为空，无法绘图")

    labels = corr.get_column(factor_name_col).cast(pl.Utf8).to_list()
    value_cols = [c for c in corr.columns if c != factor_name_col]
    if not value_cols:
        raise ValueError("corr 缺少因子相关列")

    matrix = corr.select(value_cols).to_numpy()
    if matrix.shape[0] != len(labels) or matrix.shape[1] != len(value_cols):
        raise ValueError("corr 形状与 factor_name 行数不一致")

    n_rows, n_cols = matrix.shape
    n = max(n_rows, n_cols)
    max_label_len = max(len(str(label)) for label in [*labels, *value_cols])
    layout = _correlation_heatmap_layout(n, max_label_len)
    if figsize is None:
        figsize = layout["figsize"]  # type: ignore[assignment]
    if annotate is None:
        annotate = n <= 15

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(value_cols, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("factor")
    ax.set_ylabel("factor")
    ax.set_title(title or "factor correlation matrix")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("correlation")
    cbar.ax.tick_params(labelsize=layout["tick_fontsize"])

    if annotate:
        for i in range(n_rows):
            for j in range(n_cols):
                val = matrix[i, j]
                if np.isfinite(val):
                    text = f"{val:.2f}"
                else:
                    text = "nan"
                color = "white" if np.isfinite(val) and abs(val) > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=layout["annotate_fontsize"],
                )

    ax.tick_params(axis="both", labelsize=layout["tick_fontsize"])
    ax.xaxis.label.set_fontsize(_LABEL_FONTSIZE)
    ax.yaxis.label.set_fontsize(_LABEL_FONTSIZE)
    ax.title.set_fontsize(_TITLE_FONTSIZE)

    fig.subplots_adjust(
        left=layout["left"],
        bottom=layout["bottom"],
        right=layout["right"],
    )
    if show:
        plt.show()
