"""Full, linearized and second-order loss operators for a delta-hedged short put.

Portfolio: short M puts, long M*lambda shares, lambda = delta(t, S_t) frozen over
[t, t+Delta]. Risk factor z = ln S, risk-factor change x = X_{t+Delta} (log return).
"""
import numpy as np

from black_scholes import put_price, put_delta, put_gamma, put_theta


class HedgedShortPut:
    def __init__(self, S0, K, T, r, sigma, M, t=0.0, dt=10 / 252):
        self.S0, self.K, self.T, self.r, self.sigma = S0, K, T, r, sigma
        self.M, self.t, self.dt = M, t, dt
        args = (t, S0, T, K, r, sigma)
        self.P0 = put_price(*args)
        self.lam = put_delta(*args)      # hedge ratio, held constant
        self.gamma = put_gamma(*args)
        self.theta = put_theta(*args)
        self.V0 = M * (self.lam * S0 - self.P0)

    def full(self, x):
        """l(x) = -M [ lam S_t (e^x - 1) - (P(t+dt, S_t e^x) - P(t, S_t)) ]."""
        S1 = self.S0 * np.exp(x)
        P1 = put_price(self.t + self.dt, S1, self.T, self.K, self.r, self.sigma)
        return -self.M * (self.lam * (S1 - self.S0) - (P1 - self.P0))

    def linear(self, x):
        """l_lin(x) = M theta dt: the x-coefficient M S_t (lam - delta) is zero."""
        return np.full_like(np.asarray(x, dtype=float), self.M * self.theta * self.dt)

    def quadratic(self, x):
        """l_quad(x) = M theta dt + 0.5 M gamma S_t^2 x^2."""
        return self.M * self.theta * self.dt + 0.5 * self.M * self.gamma * self.S0**2 * x**2
