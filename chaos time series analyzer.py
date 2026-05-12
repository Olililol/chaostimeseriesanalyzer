import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.matlib import repmat
from scipy import stats
from scipy.stats import linregress
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import explained_variance_score as e_variance
from sklearn.metrics import r2_score
from sklearn.neighbors import kneighbors_graph
from sklearn.feature_selection import mutual_info_regression
from scipy.signal import detrend
from statsmodels.tsa.tsatools import detrend as poly_detrend
import nolds

# Signal Analysis Methods

# Detrending a series using either linear or polynomial
def ts_detrend(signal, order):
    if order == 1:
        detrended_signal = detrend(signal)
    else:
        detrended_signal = poly_detrend(signal, order)

    trend = signal - detrended_signal
    print("\nDetrending Results:")
    print("R2", r2_score(signal, trend))
    return detrended_signal, trend

# Spectral analysis function with option for fitting slope
# and extracting Hurst exponent in the case of a power law spectrum
# used for Self-Organized Criticality markers' identification (SOC)
def pspectrum(series, ax, title, time_step = 1, fit_slope = False):
    
    # perform spectrum analysis
    ps = np.abs(np.fft.fft(series))**2
    freqs = np.fft.fftfreq(len(series), time_step)
    idx = np.argsort(freqs)
    ax.loglog(freqs[idx], ps[idx], 'k-', lw=0.6)
    if fit_slope:
        pos = freqs > 0
        log_f = np.log(freqs[pos])
        log_p = np.log(ps[pos])
        slope, intercept, r, p, se = linregress(log_f, log_p)
        print(f"Power law slope ={slope:.3f}, R^2 = {r**2:.3f}")
        ax.loglog(freqs[pos], np.exp(intercept + slope * log_f), 'r--', lw=1)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power')
    ax.set_title(title)

# Calculation of log-log histogram for case of SOC markers' identification
def loglog_hist(series, bins):
    # build histogram using bins and get relative frequencies
    hist, bin_edges = np.histogram(series, bins = bins)
    hist = hist / np.sum(hist)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    # build pandas df with class centers
    df = pd.DataFrame({'c': bin_centers, 'f': hist})
    df = df[df.f != 0]
    return np.log(df.c), np.log(df.f)


# Plot Histogram in log-log scale used for SOC markers' identification
def plot_hist(log_c, log_f, ax, fit_data = False, start = None, stop= None):
    # If a linear fit is to be applied (in case of a power law fit)
    # the fitting is done on a chosen range where power law decay has been 
    # identified
    if fit_data and start is not None and stop is not None:
        slope, intercept, r, p, se = linregress(log_c[start:stop], log_f[start:stop])
        print(f"\nSlope: {slope}\nIntercept: {intercept}\nR2: {r**2}\np-value:{p}")
        ax.scatter(log_c, log_f, c='k', s=2)
        ax.plot(log_c, intercept + slope * log_c, 'k--', lw=0.2, label='fitted line')
        ax.legend()
    else:
        ax.scatter(log_c, log_f, c='k', s=2)
    ax.set_xlabel('log(c)')
    ax.set_ylabel('log(freq)')
    ax.set_title('Histogram (absolute returns)')

# Perform R/S Analysis
def rs_analysis(data, debug_plot = False):
    H = nolds.hurst_rs(data, debug_plot=debug_plot)
    print(f"\nHurst (R/S) exponent: {H:.4f}")
    print(f"Beta (2H+1): {2*H+1:.4f}")
    return H
"""
Download stock market dataset to analyse and predict data
"""
import yfinance as yf

# Take only first 5 tickers for testing
tickers = ["AAPL", "MSFT", 'GOOGL', "AMZN", "TSLA"]
data_store = {}

for ticker in tickers:
    try:
        df = yf.download(ticker, period = "10y", auto_adjust = True, progress = False)
        if df.empty:
            print(f"No data for {ticker}")
            continue

        # Extract price series - column is 'Close' when auto_adjust = True
        if 'Close' in df.columns:
            price_series = df['Close'].values
        else:
            # Fallback: first numeric column
            price_series = df.select_dtypes(include=[np.number]).iloc[:, 0].values

        # Flatten if needed
        if price_series.ndim == 2:
            price_series = price_series.ravel()

        if len(price_series) < 2:
            print(f"Not enough price data for {ticker}")
            continue
        
        #1. Detrend price series (linear detrending)
        detrended_price, trend = ts_detrend(price_series, order=1)
        returns = np.diff(price_series) / price_series[:-1]
        returns = returns[~np.isnan(returns)]
        if len(returns) == 0:
            continue
        abs_returns = np.abs(returns)
        log_c, log_f = loglog_hist(abs_returns, bins = 50)
        H = rs_analysis(returns, debug_plot=False)

        data_store[ticker] = {
            'price': price_series,
            'trend': trend,
            'detrended': detrended_price,
            'returns': returns,
            'log_c': log_c,
            'log_f': log_f,
            'hurst': H
        }
        print(f"Processed {ticker}: Hurst = {H:.4f}")
    except Exception as e:
        print(f"Could not process {ticker}: {e}")

if not data_store:
    raise SystemExit('No data loaded.')

#Interactive plotting
fig, axes = plt.subplots(1, 3, figsize=(15,5))
plt.subplots_adjust(bottom=0.2)
current_idx = 0
ticker_list = list(data_store.keys())

def update_plot(idx):
    ticker = ticker_list[idx]
    data = data_store[ticker]

    axes[0].clear()
    axes[0].plot(data['price'], label='Original Price', alpha = 0.7)
    axes[0].plot(data['trend'], label='Linear Trend', linewidth = 2)
    axes[0].set_title(f"{ticker} - Price & Trend (H={data['hurst']:.3f})")
    axes[0].legend()

    axes[1].clear()
    pspectrum(data['detrended'], axes[1], f"{ticker} - Power Spectrum", fit_slope = True)

    axes[2].clear()
    plot_hist(data['log_c'], data['log_f'], axes[2], fit_data = True, start=0, stop=min(20, len(data['log_c'])))

    fig.suptitle(f"Stock: {ticker} | Use < > keyboard arrows to switch | Press 'q' to quit", fontsize=10)
    fig.canvas.draw_idle()

def on_key(event):
    global current_idx
    if event.key == 'right':
        current_idx = (current_idx + 1) % len(ticker_list)
        update_plot(current_idx)
    elif event.key == 'left':
        current_idx = (current_idx - 1) % len(ticker_list)
        update_plot(current_idx)
    elif event.key == 'q':
        plt.close(fig)

fig.canvas.mpl_connect('key_press_event', on_key)
update_plot(0)
plt.show()

# Print summary
print("\n--- Summary ---")
for ticker, data in data_store.items():
    print(f"{ticker}: {data['hurst']:.4f}")