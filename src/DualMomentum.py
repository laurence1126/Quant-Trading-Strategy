import numpy as np
import pandas as pd
from datetime import timedelta
import os


def data_cleaning(data: pd.DataFrame, start: int = None, end: int = None):
    if not start:
        start = 0
    if not end:
        end = len(data)
    data = data.rename(columns={data.columns[0]: "Date"})
    data = data.replace(r"^\s*$", np.nan, regex=True).iloc[start:end, :].set_index("Date").dropna(axis=1)
    return data


base_dir = os.path.dirname(os.path.dirname(__file__))
stock_price = pd.read_excel(os.path.join(base_dir, "data/S&P500 Daily Closing Price 2014-2024.xlsx"))
clean_stock_price: pd.DataFrame = data_cleaning(stock_price)
clean_stock_price.index = pd.to_datetime(clean_stock_price.index)
clean_stock_semiannually_return = clean_stock_price.resample("M").last().rolling(window=6).apply(lambda x: (x[-1] - x[0]) / x[0]).dropna()
clean_stock_semiannually_return.index = [str(i + timedelta(days=15)).rsplit("-", 1)[0] + "-01" for i in clean_stock_semiannually_return.index]

company_names = clean_stock_semiannually_return.columns.to_numpy()
long_list_output = pd.DataFrame(columns=range(1, 51))
short_list_output = pd.DataFrame(columns=range(1, 51))
for i in range(clean_stock_semiannually_return.shape[0]):
    semiannual_return = clean_stock_semiannually_return.iloc[i].values
    sorted_indices = np.argsort(-semiannual_return)
    sorted_semiannual_return = semiannual_return[sorted_indices]
    sorted_company_names = company_names[sorted_indices]
    positive_count = np.sum(sorted_semiannual_return > 0)
    negative_count = np.sum(sorted_semiannual_return < 0)
    positive_number = min(50, positive_count)
    negative_number = min(50, negative_count)
    array_shape = (50,)
    positive_insert_array = np.resize(sorted_company_names[:positive_number], array_shape)
    negative_insert_array = np.resize(sorted_company_names[-negative_number:], array_shape)
    remaining_positive_columns = 50 - positive_number
    remaining_negative_columns = 50 - negative_number
    positive_insert_array[-remaining_positive_columns:] = ""
    negative_insert_array[-remaining_negative_columns:] = ""
    if remaining_positive_columns == 0:
        long_list_output.loc[i] = sorted_company_names[:positive_number]
    else:
        long_list_output.loc[i] = positive_insert_array
    if remaining_negative_columns == 0:
        short_list_output.loc[i] = sorted_company_names[-negative_number:]
    else:
        short_list_output.loc[i] = negative_insert_array

long_list_output = long_list_output.set_index(clean_stock_semiannually_return.index)
short_list_output = short_list_output.set_index(clean_stock_semiannually_return.index)
print(long_list_output)
with pd.ExcelWriter(os.path.join(base_dir, "output/Dual Momentum Stock Selection.xlsx")) as writer:
    long_list_output.index = [str(i) for i in long_list_output.index]
    long_list_output.to_excel(writer, sheet_name="Long_list")
    short_list_output.to_excel(writer, sheet_name="Short_list")
