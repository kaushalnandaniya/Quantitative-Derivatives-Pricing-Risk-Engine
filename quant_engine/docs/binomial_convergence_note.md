# How the Binomial Model Converges to Black-Scholes

## 1. The CRR Parameterization

The Cox-Ross-Rubinstein (1979) binomial model discretizes time into N steps
of size Δt = T/N. At each step the stock price moves:

```
    S → S·u  (up)    with risk-neutral probability p
    S → S·d  (down)  with probability (1 - p)
```

Where:

```
    u = exp(σ√Δt)
    d = 1/u = exp(-σ√Δt)
    p = (exp(rΔt) - d) / (u - d)
```

The tree is **recombining**: an up-then-down move gives the same price as
down-then-up (since u·d = 1). This means after N steps there are only (N+1)
distinct prices instead of 2^N — reducing computational complexity from
exponential to O(N²) time and O(N) memory.

## 2. Why This Approximates GBM

After N steps, the log-return is:

```
    ln(S_T/S_0) = j·ln(u) + (N-j)·ln(d)
                = j·σ√Δt + (N-j)·(-σ√Δt)
                = (2j - N)·σ√Δt
```

where j is the number of up moves, distributed as Binomial(N, p).

**By the Central Limit Theorem**, as N → ∞:

```
    j ~ Binomial(N, p)  →  Normal(Np, Np(1-p))
```

Substituting back:

```
    ln(S_T/S_0)  →  Normal((r - σ²/2)T,  σ²T)
```

This is exactly the log-normal distribution assumed by Black-Scholes under
Geometric Brownian Motion:

```
    dS = rS·dt + σS·dW
```

Therefore the **CRR binomial tree is a discrete approximation to GBM**, and
option prices computed on the tree converge to Black-Scholes prices as N → ∞.

## 3. Convergence Rate

The convergence is O(1/N), but with an oscillatory component:

- **Even N** and **odd N** converge from opposite sides of the true price
- This oscillation arises because the strike price K may or may not fall
  exactly on a node of the tree
- In practice, N ≈ 200–500 steps gives accuracy within a few cents for
  typical equity options

**Richardson extrapolation** can accelerate convergence by combining prices
from N and N+1 steps.

## 4. Risk-Neutral Pricing

The probability p is **not** the real-world probability of an up move.
It is the unique probability that makes the expected discounted stock price
equal to the current price (no-arbitrage condition):

```
    S = exp(-rΔt) · [p·S·u + (1-p)·S·d]
```

Solving for p gives the CRR formula. Under this measure:
- All assets grow at the risk-free rate r in expectation
- Option prices are discounted expected payoffs
- The model is arbitrage-free by construction

This is exactly the discrete analogue of the risk-neutral measure in
continuous-time Black-Scholes theory.

## 5. American Options: Beyond Black-Scholes

The binomial tree's real power is pricing American options, where the holder
can exercise at **any** node:

```
    V = max(hold_value, exercise_value)
```

At each node during backward induction, we check whether immediate exercise
gives more value than holding. This handles:
- **American puts**: Early exercise is optimal when the stock is sufficiently
  deep in-the-money (the interest earned on K exceeds the option's time value)
- **American calls (no dividends)**: Never optimal to exercise early
  (time value is always positive), so American call = European call

No closed-form solution exists for American puts — the binomial tree
provides a **numerical** answer that converges to the true value as N → ∞.
This is why the binomial model remains the industry standard for American
option pricing alongside finite-difference and least-squares Monte Carlo methods.
