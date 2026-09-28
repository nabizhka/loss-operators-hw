# Loss Operators for a Delta-Hedged Short Put

Short M = 100 Black-Scholes puts, delta hedged with shares held fixed over a
10-day horizon. The risk factor is the log price, the risk-factor change is the
10-day log return x.

1. **Loss operators.** Full, linearized, and second-order (x^2 term only).
   The linearized loss is the constant `M * theta * Delta`, because the delta
   hedge zeroes the x coefficient.
2. **Simulation.** N = 100,000 log returns under P, pushed through all three
   operators. Histograms, VaR and ES at 95% and 99%.

## Setup and run

```bash
pip install -r requirements.txt
python src/main.py
latexmk -pdf main.tex
```

`main.py` prints the summary, writes `figures/`, and regenerates
`generated/values.tex`, which `main.tex` reads for every number it quotes.

## Layout

```
src/black_scholes.py    put price, delta, gamma, theta
src/loss_operators.py   HedgedShortPut: full / linear / quadratic loss operators
src/main.py             simulation, sanity checks, figures, LaTeX numbers
PSEUDOCODE.md           pseudocode
main.tex                write-up
```

## Notes

- The sheet's put formula has a typo (`N(d1) - 2`); the code uses
  `P = K e^{-r tau} N(-d2) - S N(-d1)` and checks put-call parity.
- Greeks are checked against finite differences on every run.
- Seed 731, so the numbers in the PDF reproduce exactly.
