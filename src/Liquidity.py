import numpy as np
import pandas as pd
import statsmodels.api as sm
import os


def data_cleaning(data: pd.DataFrame, start: int = None, end: int = None):
    if not start:
        start = 0
    if not end:
        end = len(data)
    data = data.rename(columns={data.columns[0]: "Date"})
    data = data.replace(r"^\s*$", np.nan, regex=True).iloc[start:end, :].set_index("Date").dropna(axis=1)
    return data


# using selected 100 stocks 2014-2024 data for regression
base_dir = os.path.dirname(os.path.dirname(__file__))
select_stock_volume = pd.read_excel(
    os.path.join(base_dir, "data/S&P 500 Trading Volume, Open Price 14-24.xlsx"), sheet_name="S&P 500 Trading Volume 14-24"
)
select_stock_open = pd.read_excel(
    os.path.join(base_dir, "data/S&P 500 Trading Volume, Open Price 14-24.xlsx"), sheet_name="S&P 500 Opening Price 14-24"
)
select_stock_close = pd.read_excel(os.path.join(base_dir, "data/S&P500 Daily Closing price 2014-2024.xlsx"))

# Delete empty columns
select_stock_volume = data_cleaning(select_stock_volume)
select_stock_open = data_cleaning(select_stock_open)
select_stock_close = data_cleaning(select_stock_close)
index_close = pd.read_excel(os.path.join(base_dir, "data/SPX Daily Closing Price 14-24.xlsx"), index_col=0)
print(select_stock_close.iloc[:, :15])
select_stock_average = (select_stock_close + select_stock_open) / 2
select_stock_dollar_volume = select_stock_average * select_stock_volume / 1000000
select_stock_dollar_volume = select_stock_dollar_volume.iloc[1:]
index_return = index_close.pct_change().dropna()
stock_return = select_stock_close.pct_change().dropna()

# Need to check volume and return are matched
print(index_return.shape, select_stock_dollar_volume.shape)

index_return.index = pd.to_datetime(index_return.index)
stock_return.index = pd.to_datetime(stock_return.index)
select_stock_dollar_volume.index = pd.to_datetime(select_stock_dollar_volume.index)

# Monthly Regression
target_month_list = range(1, 13)
target_year_list = range(2014, 2025)
gamma_list = []
for target_year in target_year_list:
    for target_month in target_month_list:
        if target_year == 2024 and target_month > 4:
            break
        condition = (index_return.index.month == target_month) & (index_return.index.year == target_year)
        # Makes monthly ETF return be (20,) shape
        monthly_ETF_return = index_return.loc[condition].values.flatten()
        monthly_stock_return = stock_return.loc[condition]
        monthly_select_stock_dollar_volume = select_stock_dollar_volume.loc[condition]
        monthly_gamma_mean = 0
        for stock_name in monthly_stock_return.columns:
            monthly_single_stock_return = monthly_stock_return[stock_name].to_numpy()
            monthly_single_stock_dollar_volume = monthly_select_stock_dollar_volume[stock_name].to_numpy()
            monthly_excess_return = monthly_single_stock_return - monthly_ETF_return
            # print(monthly_excess_return.shape, monthly_single_stock_return.shape,
            #       monthly_single_stock_dollar_volume.shape, monthly_ETF_return.shape)
            Y = monthly_excess_return[1:]
            X1 = monthly_single_stock_return[:-1]
            X2 = np.sign(monthly_excess_return[:-1]) * monthly_single_stock_dollar_volume[:-1]
            X = np.column_stack((X1, X2))
            X = sm.add_constant(X)
            # Check the shape of X and Y
            # print(Y.shape, X.shape)
            model = sm.OLS(Y, X)
            results = model.fit()
            beta_X2 = results.params[2]
            monthly_gamma_mean = monthly_gamma_mean + beta_X2 / len(monthly_stock_return.columns)
        gamma_list.append(monthly_gamma_mean)
        path = f"{target_year}-{target_month:02}"
        print(path, "Finished")
# Match the index and value
data_range = pd.date_range(start="2014-01", end="2024-05", freq="M").strftime("%Y-%m")
monthly_liquidity = pd.DataFrame(index=data_range)
monthly_liquidity["Monthly_liquidity"] = gamma_list
print("output!")
monthly_liquidity.to_excel(os.path.join(base_dir, "output/Liquidity Premium.xlsx"), index=True)
