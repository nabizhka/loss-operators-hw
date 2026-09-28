"""MF731 Homework: loss distribution of a delta-hedged short put.

Run from the repository root:  python src/main.py
Writes figures/ and generated/values.tex (numbers quoted in the write-up).
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from black_scholes import put_price, put_delta, put_gamma, put_theta
from loss_operators import HedgedShortPut

ROOT = Path(__file__).resolve().parents[1]
FIG, GEN = ROOT / "figures", ROOT / "generated"

# ---- parameters from the sheet ------------------------------------------------
mu, sigma, r = 0.16905, 0.4907, 0.0011888
t, T, dt = 0.0, 0.291667, 10 / 252
S0, K, M = 152.51, 170.0, 100
N, SEED = 100_000, 731


def sanity_checks():
    """Put-call parity and finite-difference greeks, to catch sign slips."""
    from scipy.stats import norm
    tau = T - t
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    call = S0 * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d1 - sigma * np.sqrt(tau))
    P = put_price(t, S0, T, K, r, sigma)
    parity_gap = (call - P) - (S0 - K * np.exp(-r * tau))
    h, k = 1e-3, 1e-6
    fd_delta = (put_price(t, S0 + h, T, K, r, sigma) - put_price(t, S0 - h, T, K, r, sigma)) / (2 * h)
    fd_gamma = (put_price(t, S0 + h, T, K, r, sigma) - 2 * P
                + put_price(t, S0 - h, T, K, r, sigma)) / h**2
    fd_theta = (put_price(t + k, S0, T, K, r, sigma) - put_price(t - k, S0, T, K, r, sigma)) / (2 * k)
    errs = {
        "parity": abs(parity_gap),
        "delta": abs(fd_delta - put_delta(t, S0, T, K, r, sigma)),
        "gamma": abs(fd_gamma - put_gamma(t, S0, T, K, r, sigma)),
        "theta": abs(fd_theta - put_theta(t, S0, T, K, r, sigma)),
    }
    for name, e in errs.items():
        assert e < 1e-4, f"{name} check failed: {e}"
    return errs


def var_es(L, alpha):
    """Empirical VaR = L_(ceil(n alpha)) as in lecture; ES = mean of losses >= VaR."""
    Ls = np.sort(L)
    var = Ls[int(np.ceil(len(L) * alpha)) - 1]
    return var, Ls[Ls >= var].mean()


def main():
    FIG.mkdir(exist_ok=True)
    GEN.mkdir(exist_ok=True)
    errs = sanity_checks()

    pf = HedgedShortPut(S0, K, T, r, sigma, M, t=t, dt=dt)

    # X_{t+dt} ~ N((mu - sigma^2/2) dt, sigma^2 dt) under P
    rng = np.random.default_rng(SEED)
    X = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(N)

    L = {"full": pf.full(X), "lin": pf.linear(X), "quad": pf.quadratic(X)}

    stats = {}
    for k, v in L.items():
        v95, e95 = var_es(v, 0.95)
        v99, e99 = var_es(v, 0.99)
        stats[k] = dict(mean=v.mean(), std=v.std(ddof=1), var95=v95, es95=e95,
                        var99=v99, es99=e99, p01=np.quantile(v, 0.01))
    err = L["quad"] - L["full"]
    corr = np.corrcoef(L["full"], L["quad"])[0, 1]
    tail = np.abs(X - X.mean()) > 2 * X.std()        # moves beyond 2 sd
    err_dn = err[tail & (X < 0)].mean()               # large drops
    err_up = err[tail & (X > 0)].mean()               # large rallies

    # ---- console summary ------------------------------------------------------
    print(f"P0={pf.P0:.4f}  delta={pf.lam:.4f}  gamma={pf.gamma:.5f}  theta={pf.theta:.4f}")
    print(f"V0={pf.V0:.2f}  shares={M * pf.lam:.2f}  lin loss={pf.M * pf.theta * dt:.4f}")
    print("sanity:", {k: f"{v:.1e}" for k, v in errs.items()})
    for k, s in stats.items():
        print(k, {a: round(b, 3) for a, b in s.items()})
    print(f"quad-full: mean={err.mean():.3f} median|.|={np.median(abs(err)):.3f} "
          f"max|.|={abs(err).max():.2f}  corr={corr:.5f}")
    print(f"mean quad-full beyond 2sd: drops={err_dn:.2f} rallies={err_up:.2f}")

    # ---- figures --------------------------------------------------------------
    colors = {"full": "#1f3b73", "quad": "#c0392b", "lin": "#2e8b57"}
    lo, hi = np.quantile(L["full"], [0.0005, 0.9995])
    bins = np.linspace(min(lo, L["lin"][0]) - 5, hi, 160)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.hist(L["full"], bins=bins, density=True, alpha=0.55, color=colors["full"], label="Full")
    ax.hist(L["quad"], bins=bins, density=True, histtype="step", lw=1.6,
            color=colors["quad"], label="Second order")
    ax.axvline(L["lin"][0], color=colors["lin"], lw=2.2, ls="--",
               label=f"Linearized (point mass at {L['lin'][0]:.2f})")
    ax.set_xlabel(r"Loss $L_{t+\Delta}$ (\$)")
    ax.set_ylabel("Density")
    ax.set_title("Simulated 10-day loss density, delta-hedged short 100 puts (N = 100,000)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "loss_histograms.png", dpi=200)
    plt.close(fig)

    xs = np.linspace(-0.45, 0.45, 400)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(xs, pf.full(xs), color=colors["full"], lw=2, label="Full $l_{[t]}(x)$")
    ax.plot(xs, pf.quadratic(xs), color=colors["quad"], lw=1.6, ls="-.",
            label=r"Second order $l^{\mathrm{quad}}_{[t]}(x)$")
    ax.plot(xs, pf.linear(xs), color=colors["lin"], lw=1.6, ls="--",
            label=r"Linearized $l^{\mathrm{lin}}_{[t]}(x)$")
    q = np.quantile(X, [0.005, 0.995])
    ax.axvspan(*q, color="grey", alpha=0.12, label="99% of simulated $x$")
    ax.set_xlabel(r"10-day log return $x$")
    ax.set_ylabel("Loss (\\$)")
    ax.set_title("Loss operators as functions of the log return")
    ax.legend(frameon=False, loc="upper center")
    fig.tight_layout()
    fig.savefig(FIG / "loss_operators.png", dpi=200)
    plt.close(fig)

    # ---- numbers for LaTeX ----------------------------------------------------
    def m(name, val, fmt="{:.2f}"):
        return f"\\newcommand{{\\{name}}}{{{fmt.format(val)}}}\n"

    out = [m("PutPrice", pf.P0, "{:.4f}"), m("PutDelta", pf.lam, "{:.4f}"),
           m("PutGamma", pf.gamma, "{:.5f}"), m("PutTheta", pf.theta, "{:.4f}"),
           m("ShareCount", M * pf.lam), m("PortValue", pf.V0),
           m("LinLoss", pf.M * pf.theta * dt), m("GammaCoef", 0.5 * M * pf.gamma * S0**2),
           m("XMean", (mu - 0.5 * sigma**2) * dt, "{:.5f}"), m("XSd", sigma * np.sqrt(dt), "{:.5f}"),
           m("ErrMean", err.mean()), m("ErrMedAbs", np.median(abs(err))),
           m("ErrMaxAbs", abs(err).max()), m("CorrFQ", corr, "{:.4f}"),
           m("ErrDown", err_dn), m("ErrUp", err_up),
           m("PdeCheck", 0.5 * M * pf.gamma * S0**2 * sigma**2 * dt)]
    names = {"full": "Full", "lin": "Lin", "quad": "Quad"}
    fields = {"mean": "Mean", "std": "Sd", "var95": "VaRa", "es95": "ESa",
              "var99": "VaRb", "es99": "ESb", "p01": "Low"}   # a = 95%, b = 99%
    for k, st in stats.items():
        for a, b in st.items():
            out.append(m(names[k] + fields[a], b))
    (GEN / "values.tex").write_text("".join(out))


if __name__ == "__main__":
    main()
