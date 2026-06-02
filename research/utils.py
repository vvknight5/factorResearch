def resolve_factor_name(factor_name: str | None, factor_col: str) -> str:
    """因子展示名；未指定时使用 factor_col。"""
    return factor_col if factor_name is None else factor_name
