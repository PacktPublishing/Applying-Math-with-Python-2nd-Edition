import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from tsdata import generate_sample_data

from matplotlib.rcsetup import cycler
plt.rc("axes", prop_cycle=cycler(c="k"))

sample_ts, test_ts = generate_sample_data(trend=0.2, undiff=True)

ts_fig, ts_ax = plt.subplots(tight_layout=True)
sample_ts.plot(ax=ts_ax, label="Observed")
ts_ax.set_title("Training time series data")
ts_ax.set_xlabel("Date")
ts_ax.set_ylabel("Value")

diffs = sample_ts.diff().dropna()

ap_fig, (acf_ax, pacf_ax) = plt.subplots(2, 1, tight_layout=True)
sm.graphics.tsa.plot_acf(diffs, ax=acf_ax)
sm.graphics.tsa.plot_pacf(diffs, ax=pacf_ax)
acf_ax.set_ylabel("Value")
acf_ax.set_xlabel("Lag")
pacf_ax.set_xlabel("Lag")
pacf_ax.set_ylabel("Value")


model = sm.tsa.ARIMA(sample_ts, order=(1,1,1))
fitted = model.fit()
print(fitted.summary())

forecast = fitted.get_forecast(steps=50).summary_frame()
print(forecast)

forecast["mean"].plot(ax=ts_ax, label="Forecast", ls="--")
ts_ax.fill_between(forecast.index, forecast["mean_ci_lower"],
                   forecast["mean_ci_upper"], alpha=0.4)


test_ts.plot(ax=ts_ax, label="Actual", ls="-.")
ts_ax.legend()

plt.show()
