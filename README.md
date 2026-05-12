# chaos time series analyzer
This code analyses long-term stock market data (10 years) for
signatures of Self-Organized Criticality (SOC) - power-law tails, long-memory 
(Hurst exponent), and 1/f-type power spectra.
Downloading price data for selected tickers, it computes detrended prices, daily returns, and generates 
interactive plots to visualise these SOC markers

The left subplot shows the Hurst exponent,
which determines the dependence of the stock prices.
If Hurst > 0.5, stock prices of that company behaves as a persistent trend.
If Hurst < 0.5, stock prices of that company behaves as a mean-reverting trend
and if Hurst is approx. 0.5, the stock prices behaves as a random walk.

The middle subplot represents the power spectrum of the detrended price. 
The straight line on log-log axes represents power-law scaling.

The right subplot represents the log-log histogram of absolute daily returns.
The black dots represent the data and black dashed line represents the power-law 
fit over the chosen range
A straight line represents heavy tails (power-law decay)

From the plots, we can interpret that AAPL and TSLA has a persistent time series,
whereby an increase in stock prices is likely to be followed by another increase
and vice versa for decreases.
On the other hand, MSFT, GOOGL, and AMZN has a hurst exponent lower than 0.5, indicating
an anti-persistent time series that does not follow a consistent trend, but reverses the
direction frequently.

