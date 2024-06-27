import os
import pandas as pd
import numpy as np
import backtrader as bt
import quantstats as qs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import Config


class BLStrategy(bt.Strategy):
    # list for tickers
    params = (("stocks", []), ("printnotify", False), ("printlog", False))

    def log(self, txt, dt=None):
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print("%s, %s" % (dt.isoformat(), txt))

    def __init__(self, weights):
        self.datafeeds = {}  # data feeds
        self.weights = weights  # weights for all stocks
        self.committed_cash = 0
        self.bar_executed = 0

        # price data and order tracking for each stock
        for i, ticker in enumerate(self.params.stocks):
            self.datafeeds[ticker] = self.datas[i]

    def notify_order(self, order):
        if self.params.printnotify:
            if order.status in [order.Submitted, order.Accepted]:
                print(f"Order for {order.size} shares of {order.data._name}" f"at {order.created.price} is {order.getstatusname()}")

            if order.status in [order.Completed]:
                if order.isbuy():
                    print(
                        f"Bought {order.executed.size} shares of {order.data._name} "
                        f"at {order.executed.price}, "
                        "cost: {order.executed.value}, "
                        "comm: {order.executed.comm}"
                    )
                elif order.issell():
                    print(
                        f"Sold {order.executed.size} shares of {order.data._name} "
                        f"at {order.executed.price}, "
                        f"cost: {order.executed.value}, "
                        f"comm: {order.executed.comm}"
                    )

            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                print(f"Order for {order.size} shares of {order.data._name} " f"at {order.created.price} is {order.getstatusname()}")

    # for each date, place orders according to the weights
    def next(self):
        date = self.data.datetime.date(0)
        weights = self.weights.loc[date.strftime("%Y-%m-%d")]

        if not self.position:
            self.log("We do not hold any positions at the moment")
        self.log(f"Total portfolio value: {self.broker.getvalue()}")

        for ticker in self.params.stocks:
            data = self.datafeeds[ticker]
            target_percent = weights[ticker]

            self.log(f"{ticker} Open: {data.open[0]}, " f"Close: {data.close[0]}, " f"Target Percent: {target_percent}")
            self.orders = self.order_target_percent(data, target=target_percent)


# define portfolio data feeds
class PandasData(bt.feeds.PandasData):
    lines = ("open", "close")
    params = (
        ("datetime", None),  # use index as datetime
        ("open", 0),  # the [0] column is open price
        ("close", 1),  # the [1] column is close price
        ("high", 0),
        ("low", 0),
        ("volume", 0),
        ("openinterest", 0),
    )


# new observer for portfolio
class PortfolioValueObserver(bt.Observer):
    lines = ("value",)
    plotinfo = dict(plot=True, subplot=True)

    def next(self):
        self.lines.value[0] = self._owner.broker.getvalue()


# backtest given prices, weights, initial cash, commission fee
def RunBacktest(stock_list, combined_df, weights_df, ini_cash, comm_fee, notify, log):
    cerebro = bt.Cerebro()  # initiate cerebro engine

    # load data feeds
    for col in stock_list:
        data = PandasData(dataname=combined_df[[col + "_open", col + "_close"]])
        cerebro.adddata(data, name=col)

    # strategy setting
    weights_df = weights_df / weights_df.sum(axis=1).values.reshape(-1, 1) * 0.9  # margin
    # set initial cash
    cerebro.broker.setcash(100000000)
    cerebro.broker.setcommission(commission=comm_fee)  # set commission
    cerebro.addstrategy(BLStrategy, weights=weights_df, stocks=stock_list, printnotify=False, printlog=False)  # set strategy
    cerebro.addobserver(PortfolioValueObserver)  # add observer
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")  # add analyzer

    # run the strategy
    results = cerebro.run()

    return results


def PortReport(returns, port_prices, spx_prices, target_return, comm, sheet_name, dirname):
    # plot the price and benchmark with drawdown
    plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(7, 1)

    # portfolio vs benchmark
    ax1 = plt.subplot(gs[0:5, :])
    ax1.plot(port_prices, label="Portfolio", color="#5aa2d4")
    ax1.plot(spx_prices, label="SPX", color="#c0c0c0")
    ax1.get_xaxis().set_visible(False)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left", frameon=False, fontsize=12, facecolor="none", edgecolor="none", labelcolor="#595959", ncol=2)

    # plot the drawdown of portfolio
    drawdown = qs.stats.to_drawdown_series(port_prices)
    ax2 = plt.subplot(gs[5:6, :])
    ax2.plot(drawdown, color="#5aa2d4")
    ax2.fill_between(drawdown.index, drawdown, 0, color="#5aa2d4", alpha=0.1)
    ax2.set_title("Drawdown")
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    file_name = f"{target_return}_comm{comm}_{sheet_name}"
    plt.suptitle(f"{file_name} Portfolio vs SPX", fontweight="bold")
    plt.savefig(f"{dirname}/../img/port_vs_spx_{file_name}.png", bbox_inches="tight")

    # quantstats report
    title = f"Target Return: {target_return}, Commission Fee: {comm} {sheet_name}"
    qs.reports.html(returns, output=f"{dirname}/../output/backtest_{file_name}_report.html", title=f"FINA4380 Portfolio {title} Report")


def data_cleaning(data: pd.DataFrame, start: int = None, end: int = None):
    if not start:
        start = 0
    if not end:
        end = len(data)

    data = data.rename(columns={data.columns[0]: "Date"})
    data = data.replace(r"^\s*$", np.nan, regex=True)
    data = data.iloc[start:end, :]
    data = data.set_index("Date")
    data = data.dropna(axis=1)
    data.index = pd.to_datetime(data.index)

    return data


# load data
def LoadData(target_return, sheet_name, dirname):
    close_prices_df = pd.read_excel(f"{dirname}/../data/S&P500 Daily Closing Price 2014-2024.xlsx", sheet_name="S&P500 2014-2024")
    open_prices_df = pd.read_excel(f"{dirname}/../data/S&P 500 Trading Volume,  Open Price 14-24.xlsx", sheet_name="S&P 500 Opening Price 14-24")
    weights_df = pd.read_excel(f"{dirname}/../output/long_SpecReturn_{target_return}.xlsx", sheet_name=sheet_name)

    # clean data
    close_prices_df = data_cleaning(close_prices_df)
    open_prices_df = data_cleaning(open_prices_df)
    weights_df = data_cleaning(weights_df)
    combined_df = open_prices_df.join(close_prices_df, lsuffix="_open", rsuffix="_close")
    combined_df = combined_df.dropna()
    combined_df = combined_df.loc[weights_df.index]
    stock_list = close_prices_df.columns

    return stock_list, combined_df, weights_df


if __name__ == "__main__":
    for i, target_return in enumerate(Config.TARGET_RETURN_LIST):
        # initialization
        dirname = os.path.dirname(__file__)
        comm_fee = int(Config.COMMISSION) / 1000

        stock_list, combined_df, weights_df = LoadData(target_return, Config.METHOD, dirname)
        results = RunBacktest(stock_list, combined_df, weights_df, Config.INIT_CASH, comm_fee, False, False)
        returns, positions, transactions, gross_lev = results[0].analyzers.getbyname("pyfolio").get_pf_items()

        returns = returns.squeeze()
        returns.index = pd.to_datetime(returns.index, format="%Y-%m-%d")
        spx_prices = pd.read_excel(f"{dirname}/../data/SPX Daily Closing Price 14-24.xlsx", index_col=0)
        spx_prices.index = pd.to_datetime(spx_prices.index, format="%Y-%m-%d")
        port_prices = (1 + returns).cumprod() * Config.INIT_CASH

        # align the date of price and returns
        returns.index = returns.index.to_period("D")
        spx_prices.index = spx_prices.index.to_period("D")
        spx_prices = spx_prices.reindex(returns.index, method="ffill")
        spx_prices.index = spx_prices.index.to_timestamp()
        returns.index = returns.index.to_timestamp()
        spx_prices = spx_prices / spx_prices.iloc[0] * Config.INIT_CASH

        PortReport(returns, port_prices, spx_prices, target_return, Config.COMMISSION, Config.METHOD, dirname)
