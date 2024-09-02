# Implementing the Black-Scholes Model in Python

## Introduction

As someone who's always been fascinated by the intersection of finance and technology, I wanted to dive into a project that brings these two worlds together. The Black-Scholes model, a cornerstone of modern financial theory, seemed like the perfect candidate. This model, widely used for pricing European call and put options, is both mathematically elegant and practically important. In this blog post, I'll walk you through my journey of implementing the Black-Scholes model in Python, using it to predict option prices for Apple stock over the course of a year, and comparing these predictions with the actual market prices.

## Understanding the Black-Scholes Model

Before we dive into the code, let’s take a moment to understand the Black-Scholes model. Developed by Fischer Black, Myron Scholes, and Robert Merton in the early 1970s, the model provides a theoretical estimate of the price of European-style options.

#### The formula for the Black-Scholes model is:

$C = S_0 N(d_1) - X e^{-rT} N(d_2)$

##### Where:

- $C$ is the price of the call option.  
- $S_0$ is the current stock price.  
- $X$ is the strike price of the option.  
- $T$ is the time to expiration (in years).  
- $r$ is the risk-free interest rate.  
- $\sigma$ (sigma) is the volatility of the stock.  
- $N(d_1)$ and $N(d_2)$ are the cumulative distribution functions of the standard normal distribution.

The model makes several assumptions, including that the stock pays no dividends during the option’s life, that markets are efficient, and that the volatility of the stock’s returns is constant over time. Despite these assumptions, the Black-Scholes model is widely used in practice and serves as a fundamental tool in options pricing.


## Project Setup

To bring this model to life, I decided to implement it in Python. The first step was to set up the environment with all the necessary libraries. Here’s what you’ll need:

- `numpy` for numerical operations.
- `pandas` for data manipulation.
- `yfinance` to fetch historical stock data.
- `scipy.stats` for statistical functions.
- `matplotlib` for data visualization.


Installing these libraries using pip:

```python
pip install numpy pandas yfinance scipy matplotlib
```

### Fetching and Preparing Data

For this project, I chose to use Apple Inc. (AAPL) stock data. Yahoo Finance provides an easy way to download historical stock data, which is where yfinance comes in handy.

Here's how I fetched the data:

```python
import yfinance as yf

stock_symbol = 'AAPL'
data = yf.download(stock_symbol, start='2023-01-01', end='2024-01-01', interval='1d')
data = data[['Close']]
```

Once I had the data, I needed to calculate the volatility of the stock, which is a crucial input for the Black-Scholes model. I calculated the annualized volatility using the stock’s daily log returns:

```python
import numpy as np

data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))
volatility = data['Log Returns'].std() * np.sqrt(252)  # Annualized volatility
```

### Implementing the Black-Scholes Model

With the data ready, it was time to implement the Black-Scholes model. The Python function below calculates the option price based on the inputs:

```python
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    S: Current stock price
    K: Option strike price
    T: Time to maturity (in years)
    r: Risk-free interest rate (annual rate)
    sigma: Volatility of the stock (annualized)
    option_type: 'call' for call option, 'put' for put option
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    return option_price
```

To apply this model, I used the stock’s closing price as the underlying price $S_0$ and the closing price on the first day as the strike price $K$. The risk-free rate $𝑟$ was assumed to be 5%, and I set the time to maturity $𝑇$ to one year.

### Analyzing the Results

The key part of this project was comparing the Black-Scholes predicted prices with the actual stock prices over the year. I created a new column in the DataFrame for the predicted prices:
