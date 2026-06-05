"""多因子正交化：将新因子对基准因子做截面回归并取残差。"""

from __future__ import annotations

import polars as pl

from research.expr import _is_finite


def _require_columns(df: pl.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"df 缺少必需列: {', '.join(missing)}")


def _solve_linear_system(a: list[list[float]], b: list[float]) -> list[float] | None:
    """高斯消元求解 Ax = b；奇异时返回 None。"""
    n = len(b)
    if n == 0 or len(a) != n or any(len(row) != n for row in a):
        return None

    mat = [row[:] + [b[i]] for i, row in enumerate(a)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(mat[r][col]))
        if abs(mat[pivot][col]) < 1e-12:
            return None
        if pivot != col:
            mat[col], mat[pivot] = mat[pivot], mat[col]

        pivot_val = mat[col][col]
        for row in range(col + 1, n):
            factor = mat[row][col] / pivot_val
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                mat[row][j] -= factor * mat[col][j]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        rhs = mat[i][n]
        for j in range(i + 1, n):
            rhs -= mat[i][j] * x[j]
        diag = mat[i][i]
        if abs(diag) < 1e-12:
            return None
        x[i] = rhs / diag
    return x


def _cross_section_ols_residuals(
    y: list[float],
    x_cols: list[list[float]],
) -> list[float] | None:
    """带截距的截面多元 OLS，返回残差；样本不足或矩阵奇异时返回 None。"""
    n = len(y)
    k = len(x_cols)
    if n == 0 or k == 0 or any(len(col) != n for col in x_cols):
        return None
    if n <= k:
        return None

    p = k + 1
    xtx = [[0.0] * p for _ in range(p)]
    xty = [0.0] * p

    for i in range(n):
        row = [1.0, *[x_cols[j][i] for j in range(k)]]
        for a in range(p):
            xty[a] += row[a] * y[i]
            for b in range(p):
                xtx[a][b] += row[a] * row[b]

    beta = _solve_linear_system(xtx, xty)
    if beta is None:
        return None

    return [
        y[i] - sum(beta[a] * (1.0 if a == 0 else x_cols[a - 1][i]) for a in range(p))
        for i in range(n)
    ]


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
        tmp_col = "_orth_tmp"
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


def orthogonalize_factor(
    df: pl.DataFrame,
    *,
    base_factors: list[str],
    new_factor: str,
    ret_col: str,
    output_col: str | None = None,
    date_col: str = "trade_date",
    stock_col: str = "stock_code",
) -> pl.DataFrame:
    """将 ``new_factor`` 对每个交易日截面正交化到 ``base_factors`` 张成的子空间。

    在每个 ``date_col`` 分组内，对 ``new_factor`` 关于 ``base_factors`` 做带截距的
    截面 OLS，输出回归残差作为正交化因子。仅使用 ``new_factor`` 与全部 ``base_factors``
    均为有限值的样本估计回归；无效行输出 ``null``。

    参数
    ----
    base_factors:
        基准因子列名列表，至少 1 列。
    new_factor:
        待正交化因子列名，不得出现在 ``base_factors`` 中。
    ret_col:
        收益率列名（面板必需列，用于数据约定校验）。
    output_col:
        输出列名，默认 ``{new_factor}_orth``；若与 ``new_factor`` 同名则替换原因子列。
    """
    if not base_factors:
        raise ValueError("base_factors 至少包含 1 列")
    if new_factor in base_factors:
        raise ValueError("new_factor 不得出现在 base_factors 中")

    required = (new_factor, ret_col, date_col, stock_col, *base_factors)
    _require_columns(df, required)

    out_col = output_col or f"{new_factor}_orth"

    valid = _is_finite(new_factor)
    for col in base_factors:
        valid = valid & _is_finite(col)

    ready = df.filter(valid).select(
        pl.col(date_col),
        pl.col(stock_col),
        pl.col(new_factor),
        *[pl.col(c) for c in base_factors],
    )

    if ready.is_empty():
        return _attach_output_column(
            df,
            None,
            out_col=out_col,
            factor_col=new_factor,
            date_col=date_col,
            stock_col=stock_col,
        )

    transform_parts: list[pl.DataFrame] = []

    for date_val, sub in ready.group_by(date_col, maintain_order=True):
        d = date_val[0]
        y = sub.get_column(new_factor).to_list()
        x_cols = [sub.get_column(c).to_list() for c in base_factors]
        residuals = _cross_section_ols_residuals(y, x_cols)
        if residuals is None:
            continue

        transform_parts.append(
            pl.DataFrame(
                {
                    date_col: [d] * len(residuals),
                    stock_col: sub.get_column(stock_col).to_list(),
                    out_col: residuals,
                },
                schema={
                    date_col: df.schema[date_col],
                    stock_col: df.schema[stock_col],
                    out_col: pl.Float64,
                },
            )
        )

    transformed_df = pl.concat(transform_parts) if transform_parts else None
    return _attach_output_column(
        df,
        transformed_df,
        out_col=out_col,
        factor_col=new_factor,
        date_col=date_col,
        stock_col=stock_col,
    )
