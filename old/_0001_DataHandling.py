# build_parquet.py  — creates Data.parquet from DATA_QARM.xlsx
import pandas as pd
import polars as pl
from pathlib import Path

SRC = Path("DATA_QARM.xlsx")
OUT = Path("Data.parquet")

# 1) Read Excel (day-first format like 20.10.2015)
df = pd.read_excel(SRC)
df.rename(columns={df.columns[0]: "Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date")

# 2) Map ETF names -> GICS sector labels (+ S&P 500 benchmark)
rename_map = {
    "FIRST TRUST MATERIALS ALPHADEX FUND": "Materials",
    "VANGUARD CONSUMER STAPLES INDEX FUND ETF": "Consumer Staples",
    "ISHARES US ENERGY": "Energy",
    "FIDELITY MSCI INDUSTRIALS INDEX ETF": "Industrials",
    "ISHARES US CONSUMER DISCRETIONARY ETF": "Consumer Discretionary",
    "ISHARES US HEALTHCARE ETF": "Health Care",
    "FIDELITY MSCI FINANCIALS INDEX ETF": "Financials",
    "VANGD.ITECH.IX.ETF": "Information Technology",
    "ISHARES US TELECOM.": "Communication Services",
    "ISHARES US UTILITIES ETF": "Utilities",
    "VANGUARD REAL ESTATE INDEX FUND ETF": "Real Estate",
    "ISHARES CORE S&P 500 ETF": "S&P 500",
}
df = df.rename(columns=rename_map)

# 3) Keep only known columns
cols = ["Date"] + list(rename_map.values())
df = df[cols]

# 4) Tidy long format and compute daily returns by asset
long = (
    df.melt(id_vars="Date", var_name="asset", value_name="price")
      .dropna(subset=["price"])
      .drop_duplicates(subset=["Date", "asset"])
      .sort_values(["asset", "Date"])
)
long["ret"] = long.groupby("asset")["price"].pct_change()

# 5) Write compressed Parquet (fast for analytics)
pl_df = pl.from_pandas(long)
pl_df.write_parquet(OUT.as_posix(), compression="zstd", statistics=True)
print(f"Saved {OUT.resolve()} with {pl_df.height} rows, {pl_df.width} cols.")
