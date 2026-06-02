import polars as pl

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
