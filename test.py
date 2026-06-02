import polars as pl
from factorresearch import evaluate_factor, factor_group_backtest

df = pl.read_csv("./data/factor_data.csv",schema_overrides={"trade_date": pl.Date, "stock_code": pl.String})
result = evaluate_factor(df, ret_col="fut_ret")

print(df)
print(result)