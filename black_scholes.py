"""Black-Scholes European put and its greeks (delta, gamma, theta)."""
import numpy as np
from scipy.stats import norm


def d1_d2(tau, x, K, r, sigma):
    d1 = (np.log(x / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return d1, d1 - sigma * np.sqrt(tau)


def put_price(t, x, T, K, r, sigma):
    """P^BS(t, x) = K e^{-r tau} N(-d2) - x N(-d1)."""
    tau = T - t
    d1, d2 = d1_d2(tau, x, K, r, sigma)
    return K * np.exp(-r * tau) * norm.cdf(-d2) - x * norm.cdf(-d1)


def put_delta(t, x, T, K, r, sigma):
    d1, _ = d1_d2(T - t, x, K, r, sigma)
    return norm.cdf(d1) - 1.0


def put_gamma(t, x, T, K, r, sigma):
    tau = T - t
    d1, _ = d1_d2(tau, x, K, r, sigma)
    return norm.pdf(d1) / (x * sigma * np.sqrt(tau))


def put_theta(t, x, T, K, r, sigma):
    """Partial derivative in calendar time t (per year)."""
    tau = T - t
    d1, d2 = d1_d2(tau, x, K, r, sigma)
    return (-sigma * x * norm.pdf(d1) / (2.0 * np.sqrt(tau))
            + K * r * np.exp(-r * tau) * norm.cdf(-d2))
