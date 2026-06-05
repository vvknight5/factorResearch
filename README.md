# factorresearch

基于 [Polars](https://pola.rs/) 的 A 股因子研究工具包：预处理、非线性因子线性化、IC 评价、分组回测与可视化。

## 安装

```bash
pip install factorresearch
# 或在本仓库目录
uv sync
```

要求 Python >= 3.14。

## 快速开始

```python
import polars as pl
from factorresearch.expr import (
    neutralize_market_cap_expr,
    standardize_cross_section_expr,
)

df = pl.read_csv("factor_data.csv", schema_overrides={"trade_date": pl.Date})

# 预处理（表达式，保留原表其余列）
df = df.with_columns(
    standardize_cross_section_expr("mom_20").alias("mom_20_z"),
    neutralize_market_cap_expr("mom_20_z", "log_mv").alias("mom_20_neu"),
)

# 评价与回测
metrics = evaluate_factor(df, factor_col="mom_20", ret_col="fut_ret", factor_name="20日动量")
annual_ic = evaluate_factor_by_period(df, period="year", factor_col="mom_20", ret_col="fut_ret")
daily_ic = plot_factor_ic(df, factor_col="mom_20", ret_col="fut_ret", factor_name="20日动量")
perf = factor_group_backtest(df, n=5, factor_col="mom_20", ret_col="fut_ret", factor_name="20日动量")
```

---

## 数据约定

### 面板表结构

多数函数要求输入为 **股票 × 交易日** 面板 `pl.DataFrame`，常用列如下：

| 列名 | 说明 |
|------|------|
| `stock_code` | 股票代码 |
| `trade_date` | 交易日（可通过 `date_col` / `stock_col` 自定义列名） |
| `factor` / 自定义 | 因子值（`factor_col`） |
| `ret` / 自定义 | 收益率（评价、回测时，`ret_col`） |
| `log_mv` | 对数市值（市值中性化时） |

### `factor_name`

所有因子相关函数均支持可选参数 `factor_name`：

- 未指定时，默认等于 `factor_col`（数据列名）。
- 用于在 **结果表** 或 **图表标题** 中标识因子（可与列名不同，如 `"20日动量"`）。
- 评价、回测、预处理的结果中会包含 `factor_name` 列，便于多因子结果合并。

### 无效值处理

| 模块 | 处理方式 |
|------|----------|
| 预处理（中性化、标准化） | 仅用有限值（非 null、非 NaN、非 Inf）估计截面统计；无效行输出因子为 `null`；市值中性化要求因子与对数市值 **同时** 有效 |
| 分段线性回归 | 估计映射时剔除因子、收益率为无效值的样本；尚无足够回溯窗口的行输出为 `null` |
| 因子评价 | 计算 IC 前剔除因子、收益率为 `null` 的样本 |
| 分组回测 | 分组前剔除因子、收益率为 `null` 的样本 |

---

## 公开 API

推荐从包根导入：

```python
# DataFrame 封装
from factorresearch import (
    evaluate_factor,
    evaluate_factor_by_period,
    plot_factor_ic,
    factor_group_backtest,
    neutralize_market_cap,
    standardize_cross_section,
    piecewise_linear_regression,
)

# Polars 表达式 / pipe 步骤
from factorresearch.expr import (
    neutralize_market_cap_expr,
    standardize_cross_section_expr,
    factor_group_expr,
    factor_percentile_expr,
    piecewise_linear_regression_expr,
)
```

绘图函数从 `plotting` 子模块导入：

```python
from plotting import hist_plot, plot_group_backtest, plot_ic, plot_quantile_bucket_curve
```

---

## 因子预处理

预处理提供两类接口，分属不同模块：

| 类型 | 模块 | 函数 | 用途 |
|------|------|------|------|
| **表达式**（推荐） | `research.expr` | `standardize_cross_section_expr`、`neutralize_market_cap_expr` | 在 `with_columns` 中就地处理 |
| **pipe 步骤** | `research.expr` | `piecewise_linear_regression_expr` | 在 `df.pipe(...)` 中链式线性化 |
| **DataFrame** | `research.preprocessing` | `standardize_cross_section`、`neutralize_market_cap` | `select` 关键列并附加 `factor_name` |
| **DataFrame** | `research.linearization` | `piecewise_linear_regression` | 原表追加或替换转换后的因子列 |

表达式已内置按 `date_col` 的截面分组（`.over(date_col)`），无需再链式调用 `.over()`。

---

### `research.expr` — Polars 表达式

#### `standardize_cross_section_expr`

截面 **z-score**：\(z = (x - \mu) / \sigma\)。

```python
from factorresearch.expr import standardize_cross_section_expr

standardize_cross_section_expr(
    factor_col: str, *, date_col: str = "trade_date"
) -> pl.Expr
```

```python
df.with_columns(standardize_cross_section_expr("mom_20").alias("mom_20_z"))
```

#### `neutralize_market_cap_expr`

截面 **市值中性化**：对数市值回归残差。

```python
from factorresearch.expr import neutralize_market_cap_expr

neutralize_market_cap_expr(
    factor_col: str, log_mv_col: str, *, date_col: str = "trade_date"
) -> pl.Expr
```

```python
df.with_columns(neutralize_market_cap_expr("mom_20", "log_mv").alias("mom_20_neu"))
```

#### `factor_group_expr` / `factor_percentile_expr`

截面 **等分分组** 与 **百分位** 辅助表达式（分段线性回归内部亦使用）。

```python
from factorresearch.expr import factor_group_expr, factor_percentile_expr

df.with_columns(
    factor_group_expr("factor", 100).alias("group"),
    factor_percentile_expr("factor").alias("pct"),
)
```

#### `piecewise_linear_regression_expr`

**分段线性回归** 的 pipe 封装，参数与 `piecewise_linear_regression` 一致。算法需跨日历史面板，无法写成单行 `pl.Expr`。

```python
from factorresearch.expr import piecewise_linear_regression_expr

# 推荐：Polars pipe 风格
panel = (
    df.with_columns(standardize_cross_section_expr("factor").alias("factor_z"))
    .pipe(
        piecewise_linear_regression_expr,
        "factor_z",
        "fut_ret",
        output_col="factor_plr",
    )
)

# 亦可先绑定参数再 pipe
panel = df.pipe(piecewise_linear_regression_expr("factor", "fut_ret", output_col="factor"))
```

---

### `research.preprocessing` — DataFrame 封装

```python
from factorresearch import standardize_cross_section, neutralize_market_cap

# 仅返回 stock_code、trade_date、处理后因子、factor_name
out = standardize_cross_section(df, factor_col="mom_20", factor_name="20日动量")
```

---

## 因子线性化

### `piecewise_linear_regression`

对非线性因子做 **分段线性回归线性化**：在估计窗口内拟合「因子截面分位 → 组超额收益」映射，并将映射应用于当日截面因子。

对交易日 `t`，映射仅使用 `[t - decay - lookback_days + 1, t - decay]` 的数据估计，并应用于 `t` 日截面。返回输入 `df` 并追加（或替换）转换后的因子列。

```python
from factorresearch import piecewise_linear_regression

piecewise_linear_regression(
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
) -> pl.DataFrame
```

**输入必需列**：`stock_code`、`date_col`、`factor_col`、`ret_col`

**参数说明**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `output_col` | `{factor_col}_plr` | 输出列名；若与 `factor_col` 同名则 **替换** 原因子列 |
| `bench_ret_col` | `None` | 基准收益列；未指定时对 `ret_col` 按交易日等权平均 |
| `n_groups` | `100` | 估计映射时的截面分组数 M |
| `lookback_days` | `120` | 估计各组超额收益的回溯交易日数 N |
| `decay` | `2` | 映射估计的最晚滞后交易日数 |
| `trading_days` | `252` | 年化换算使用的交易日数 |

**无效值与输出**：

- 估计映射时剔除因子、收益率为无效值的样本
- 尚无足够回溯窗口的行（前 `decay + lookback_days - 1` 个交易日）输出为 `null`
- `output_col` 与 `factor_col` 不同时追加新列；相同时原地替换原因子列

```python
# 追加新列（默认 factor_plr）
df = piecewise_linear_regression(df, factor_col="factor", ret_col="fut_ret")

# 原地替换 factor 列
df = piecewise_linear_regression(
    df, factor_col="factor", ret_col="fut_ret", output_col="factor",
)
```

---

## 因子评价

### `evaluate_factor`

按交易日计算因子与收益率的 **Pearson IC**、**Spearman rank IC**，并汇总全样本统计量（单行结果）。

```python
evaluate_factor(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    ret_col: str = "ret",
    date_col: str = "trade_date",
    factor_name: str | None = None,
) -> pl.DataFrame
```

**输入必需列**：`stock_code`、`date_col`、`factor_col`、`ret_col`

**返回列**（1 行）：

| 列名 | 说明 |
|------|------|
| `factor_name` | 因子标识 |
| `ic_mean` | IC 均值 |
| `ic_std` | IC 标准差 |
| `ic_max` / `ic_min` | IC 最大 / 最小 |
| `ic_t_stat` | IC 均值 t 统计量：\(\bar{ic} / (s / \sqrt{n})\) |
| `icir` | 信息比率：\(\bar{ic} / s\) |
| `rank_ic_mean` | rank IC 均值 |
| `rank_ic_t_stat` | rank IC 均值 t 统计量 |

---

### `evaluate_factor_by_period`

按 **年** 或 **月** 汇总 IC / rank IC 统计量（多行结果）。统计列与 `evaluate_factor` 相同，另含 `period` 列。

```python
evaluate_factor_by_period(
    df: pl.DataFrame,
    *,
    period: Literal["year", "month"] = "year",
    factor_col: str = "factor",
    ret_col: str = "ret",
    date_col: str = "trade_date",
    factor_name: str | None = None,
) -> pl.DataFrame
```

**输入必需列**：与 `evaluate_factor` 相同

**参数说明**：

| 参数 | 说明 |
|------|------|
| `period` | `"year"` 时 `period` 为年份整数；`"month"` 时为 `"YYYY-MM"` 字符串 |

**返回列**（每个 period 一行）：`period`、`factor_name`，以及 `ic_mean`、`ic_std`、`ic_max`、`ic_min`、`ic_t_stat`、`icir`、`rank_ic_mean`、`rank_ic_t_stat`

```python
annual_ic = evaluate_factor_by_period(df, period="year", ret_col="fut_ret")
monthly_ic = evaluate_factor_by_period(df, period="month", ret_col="fut_ret", factor_name="20日动量")
```

---

### `plot_factor_ic`

计算 **Pearson 日度 IC** 并绘制 IC 序列图，返回含累计 IC 的日度表。

```python
plot_factor_ic(
    df: pl.DataFrame,
    *,
    factor_col: str = "factor",
    ret_col: str = "ret",
    date_col: str = "trade_date",
    factor_name: str | None = None,
    figsize: tuple[float, float] = (14, 6.5),
    show: bool = True,
) -> pl.DataFrame
```

**输入必需列**：与 `evaluate_factor` 相同

**返回列**：`date_col`、`ic`、`cum_ic`（日 IC 累计和）

**图形说明**（单图、双 y 轴）：

| 元素 | 说明 |
|------|------|
| 横轴 | 交易日 |
| 左 y 轴（`C0`） | 日 IC 柱形图；虚线标平均 IC；legend 显示 `mean IC` |
| 右 y 轴（`C3`） | 日 IC 累计和折线；虚线标 y=0 |
| 标题 | 显示 IC 序列 t 统计量（与 `ic_t_stat` 计算一致） |

```python
from factorresearch import plot_factor_ic

daily_ic = plot_factor_ic(df, ret_col="fut_ret", factor_name="20日动量")
```

---

## 分组回测

### `factor_group_backtest`

每个交易日内按因子值 **截面分 N 组**（`qcut`），组内等权平均收益率作为组日收益，并计算累计收益与绩效指标。默认弹出分组回测图。

```python
factor_group_backtest(
    df: pl.DataFrame,
    n: int,
    *,
    factor_col: str = "factor",
    ret_col: str = "ret",
    date_col: str = "trade_date",
    factor_name: str | None = None,
    trading_days: int = 252,
    figsize: tuple[float, float] = (14, 5),
    show: bool = True,
) -> pl.DataFrame
```

**输入必需列**：`stock_code`、`date_col`、`factor_col`、`ret_col`

**参数说明**：

| 参数 | 说明 |
|------|------|
| `n` | 分组数，须 >= 1；`group=1` 为因子值最小组 |
| `trading_days` | 年化所用交易日数（默认 252） |
| `show` | 是否调用 `plot_group_backtest` 绘图 |

**收益计算**：单利——累计收益 = 日收益之和；年化收益 = 日均收益 × `trading_days`。

**返回列**（每组一行）：

| 列名 | 说明 |
|------|------|
| `factor_name` | 因子标识 |
| `group` | 组号 1 … N |
| `cum_return` | 累计收益 |
| `annual_return` | 年化收益 |
| `annual_vol` | 年化波动率 |
| `sharpe` | 年化夏普（收益 / 波动） |
| `max_drawdown` | 最大回撤（基于累计收益曲线） |
| `mean_daily_return` | 日均收益 |
| `win_rate` | 日收益为正的比例 |
| `profit_loss_ratio` | 盈亏比（平均盈利日收益 / 平均亏损日收益绝对值） |

---

## 绘图

### `plot_group_backtest`

绘制分组回测双子图：左图为各组累计收益曲线，右图为各组最终累计收益柱形图。一般由 `factor_group_backtest(show=True)` 自动调用。

```python
plot_group_backtest(
    cum: pl.DataFrame,
    perf: pl.DataFrame,
    n: int,
    *,
    date_col: str = "trade_date",
    figsize: tuple[float, float] = (14, 5),
    factor_name: str | None = None,
) -> None
```

- `cum`：含 `date_col`、`group`、`cum_ret` 的累计收益序列
- `perf`：含 `group`、`cum_return` 等的绩效表（通常来自 `factor_group_backtest` 的返回值）
- `factor_name`：显示在子图标题前缀

---

### `hist_plot`

绘制指定列的直方分布图。

```python
hist_plot(df, col: str, bins: int = 50, alpha: float = 0.7) -> None
```

---

### `plot_quantile_bucket_curve`

按 `x_col` 全样本分位数切 `N` 组，绘制组内 `x` 均值 vs `y` 均值曲线（`y` 为 0/1 时表示占比）。

```python
plot_quantile_bucket_curve(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    N: int,
    *,
    figsize: tuple[float, float] = (8, 4),
    show: bool = True,
    title_msg: str = "",
) -> pl.DataFrame
```

**返回**：含 `x_mean`、`y_rate` 的聚合表。

---

### `plot_ic`

基于已计算的日度 IC 序列绘图（底层接口，一般由 `plot_factor_ic` 调用）。

```python
plot_ic(
    daily: pl.DataFrame,
    *,
    date_col: str = "trade_date",
    ic_col: str = "ic",
    factor_name: str | None = None,
    figsize: tuple[float, float] = (14, 6.5),
    show: bool = True,
) -> None
```

- `daily`：含 `date_col`、`ic_col` 的日度 IC 表
- 图形规格与 `plot_factor_ic` 相同

---

## 典型工作流

```python
import polars as pl
from factorresearch import (
    neutralize_market_cap,
    standardize_cross_section,
    evaluate_factor,
    evaluate_factor_by_period,
    plot_factor_ic,
    factor_group_backtest,
    piecewise_linear_regression,
)

raw = pl.read_csv("factor_data.csv", schema_overrides={"trade_date": pl.Date})

# 1. 预处理（链式表达式 + 线性化）
from factorresearch.expr import (
    neutralize_market_cap_expr,
    standardize_cross_section_expr,
    piecewise_linear_regression_expr,
)

panel = (
    raw.with_columns(
        standardize_cross_section_expr("raw_factor").alias("raw_factor"),
    )
    .with_columns(
        neutralize_market_cap_expr("raw_factor", "log_mv").alias("raw_factor"),
    )
    .pipe(
        piecewise_linear_regression_expr,
        "raw_factor",
        "fut_ret",
        output_col="raw_factor",
    )
)

# 2. 评价与回测（过滤 null 样本）
valid = panel.filter(pl.col("raw_factor").is_not_null())
ic_table = evaluate_factor(valid, factor_col="raw_factor", ret_col="fut_ret")
annual_ic = evaluate_factor_by_period(valid, period="year", factor_col="raw_factor", ret_col="fut_ret")
daily_ic = plot_factor_ic(valid, factor_col="raw_factor", ret_col="fut_ret", show=False)
perf = factor_group_backtest(valid, n=10, factor_col="raw_factor", ret_col="fut_ret", show=False)

# 3. 多因子结果合并（均含 factor_name 列）
# all_ic = pl.concat([ic_table, other_ic_table])
```

---

## 模块结构

```
factorresearch/          # 包入口（re-export）
research/
  expr.py                # Polars 表达式与 pipe 步骤
  preprocessing.py       # 预处理 DataFrame 封装
  linearization.py       # 分段线性回归
  evaluation.py          # IC 评价
  group_backtest.py      # 分组回测
plotting/
  plots.py               # 绘图
```

## 依赖

- polars >= 1.40.1
- matplotlib >= 3.10.9
