# factorresearch

基于 [Polars](https://pola.rs/) 的 A 股因子研究工具包：预处理、正交化、IC 评价、因子相关、分组回测与可视化。

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
| 因子正交化 | 仅用 ``new_factor`` 与全部 ``base_factors`` 均有限的样本估计截面回归；无效行输出 ``null`` |
| 因子评价 / 因子相关 | 计算 IC 或截面相关前剔除无效样本 |
| 分组回测 | 分组前剔除因子、收益率为 `null` 的样本 |

---

## 公开 API

推荐从包根导入：

```python
# DataFrame 封装
from factorresearch import (
    evaluate_factor,
    evaluate_factor_by_period,
    factor_correlation_matrix,
    factor_group_backtest,
    neutralize_market_cap,
    orthogonalize_factor,
    plot_factor_correlation_matrix,
    plot_factor_ic,
    standardize_cross_section,
)

# Polars 表达式（推荐在 with_columns 中使用）
from factorresearch.expr import (
    neutralize_market_cap_expr,
    orthogonalize_factor_expr,
    standardize_cross_section_expr,
)
```

其他绘图函数从 `plotting` 子模块导入：

```python
from plotting import hist_plot, plot_group_backtest, plot_ic, plot_quantile_bucket_curve
```

也可从包根导入相关矩阵热图：

```python
from factorresearch import plot_factor_correlation_matrix
```

---

## 因子预处理

预处理提供两类接口，分属不同模块：

| 类型 | 模块 | 函数 | 用途 |
|------|------|------|------|
| **表达式**（推荐） | `research.expr` | `standardize_cross_section_expr`、`neutralize_market_cap_expr` | 在 `with_columns` 中就地处理 |
| **DataFrame** | `research.preprocessing` | `standardize_cross_section`、`neutralize_market_cap` | `select` 关键列并附加 `factor_name` |

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

---

### `research.preprocessing` — DataFrame 封装

```python
from factorresearch import standardize_cross_section, neutralize_market_cap

# 仅返回 stock_code、trade_date、处理后因子、factor_name
out = standardize_cross_section(df, factor_col="mom_20", factor_name="20日动量")
```

---

## 因子正交化

将新因子对每个交易日截面正交化到基准因子张成的子空间：对 ``new_factor`` 关于 ``base_factors`` 做带截距的截面 OLS，输出回归残差。

### `orthogonalize_factor`

```python
orthogonalize_factor(
    df: pl.DataFrame,
    *,
    base_factors: list[str],
    new_factor: str,
    ret_col: str,
    output_col: str | None = None,
    date_col: str = "trade_date",
    stock_col: str = "stock_code",
) -> pl.DataFrame
```

**输入必需列**：`stock_code`、`date_col`、`ret_col`、`new_factor`、全部 `base_factors`

**参数说明**：

| 参数 | 说明 |
|------|------|
| `base_factors` | 基准因子列名列表，至少 1 列 |
| `new_factor` | 待正交化因子，不得出现在 `base_factors` 中 |
| `ret_col` | 收益率列（面板约定校验） |
| `output_col` | 默认 `{new_factor}_orth`；与 `new_factor` 同名时替换原因子列 |

**返回**：原表并追加（或替换）正交化因子列。

```python
from factorresearch import orthogonalize_factor

panel = orthogonalize_factor(
    df,
    base_factors=["mom_20", "ep"],
    new_factor="roe",
    ret_col="fut_ret",
    output_col="roe_orth",
)
```

### `orthogonalize_factor_expr`

| 场景 | 用法 |
|------|------|
| 单基准因子 | `orthogonalize_factor_expr("roe", "mom_20")` 返回 `pl.Expr`，用于 `with_columns` |
| 多基准因子 | `df.pipe(orthogonalize_factor_expr, "roe", ["mom_20", "ep"], "fut_ret")` |

```python
from factorresearch.expr import orthogonalize_factor_expr

# 单基准
panel = df.with_columns(
    orthogonalize_factor_expr("roe", "mom_20").alias("roe_orth"),
)

# 多基准
panel = df.pipe(
    orthogonalize_factor_expr,
    "roe",
    ["mom_20", "ep"],
    "fut_ret",
    output_col="roe_orth",
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

### `factor_correlation_matrix`

计算因子间**平均截面相关矩阵**：每个交易日分别计算因子对的截面相关，再对时间取平均。

```python
factor_correlation_matrix(
    df: pl.DataFrame,
    factor_cols: list[str],
    *,
    date_col: str = "trade_date",
    method: Literal["pearson", "spearman"] = "pearson",
) -> pl.DataFrame
```

**输入必需列**：`date_col`、全部 `factor_cols`

**返回**：方阵，`factor_name` 为行标签，其余列为各因子名；对角线为 1，矩阵对称。

```python
from factorresearch import factor_correlation_matrix

corr = factor_correlation_matrix(
    df,
    ["mom_20", "ep", "roe"],
    method="spearman",
)
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

### `plot_factor_correlation_matrix`

绘制因子相关矩阵热图，输入通常为 `factor_correlation_matrix` 的返回值。

```python
plot_factor_correlation_matrix(
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
) -> None
```

未指定 `figsize` 时，会根据因子数量与标签长度自动调整图像尺寸、字号与边距；因子数 > 15 时默认关闭格内数值标注。

```python
from factorresearch import factor_correlation_matrix, plot_factor_correlation_matrix

corr = factor_correlation_matrix(df, ["mom_20", "ep", "roe"])
plot_factor_correlation_matrix(corr, title="因子截面相关")
```

---

## 典型工作流

```python
import polars as pl
from factorresearch import (
    evaluate_factor,
    evaluate_factor_by_period,
    factor_correlation_matrix,
    factor_group_backtest,
    orthogonalize_factor,
    plot_factor_correlation_matrix,
    plot_factor_ic,
    standardize_cross_section,
)
from factorresearch.expr import (
    neutralize_market_cap_expr,
    orthogonalize_factor_expr,
    standardize_cross_section_expr,
)

raw = pl.read_csv("factor_data.csv", schema_overrides={"trade_date": pl.Date})

# 1. 预处理（链式表达式）
panel = raw.with_columns(
    standardize_cross_section_expr("raw_factor").alias("raw_factor"),
).with_columns(
    neutralize_market_cap_expr("raw_factor", "log_mv").alias("raw_factor"),
)

# 2. 新因子正交化到库内因子
panel = panel.pipe(
    orthogonalize_factor_expr,
    "new_factor",
    ["raw_factor"],
    "fut_ret",
    output_col="new_factor_orth",
)

# 3. 评价与回测
ic_table = evaluate_factor(panel, factor_col="new_factor_orth", ret_col="fut_ret")
annual_ic = evaluate_factor_by_period(
    panel, period="year", factor_col="new_factor_orth", ret_col="fut_ret",
)
daily_ic = plot_factor_ic(panel, factor_col="new_factor_orth", ret_col="fut_ret", show=False)
perf = factor_group_backtest(panel, n=10, factor_col="new_factor_orth", ret_col="fut_ret", show=False)

# 4. 多因子相关与可视化
corr = factor_correlation_matrix(panel, ["raw_factor", "new_factor", "new_factor_orth"])
plot_factor_correlation_matrix(corr, show=False)
```

---

## 模块结构

```
factorresearch/          # 包入口（re-export）
research/
  expr.py                # Polars 表达式（with_columns / pipe）
  preprocessing.py       # DataFrame 封装
  orthogonalization.py   # 因子正交化
  evaluation.py          # IC 评价、因子相关
  linearization.py       # 分段线性回归
  group_backtest.py      # 分组回测
plotting/
  plots.py               # 绘图（含相关矩阵热图）
```

## 依赖

- polars >= 1.40.1
- matplotlib >= 3.10.9
