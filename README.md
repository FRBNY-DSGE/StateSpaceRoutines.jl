# State-Space Routines

This package implements some common routines for state-space models with the
following representation:

```
z_{t+1} = CCC + TTT*z_t + RRR*ϵ_t          (transition equation)
y_t     = DD  + ZZ*z_t  + MM*ϵ_t  + η_t    (measurement equation)

ϵ_t ∼ N(0, QQ)
η_t ∼ N(0, EE)
```

The provided algorithms are:

- Kalman filter (`kalman_filter`)
- Kalman smoothers:
  + `hamilton_smoother`: James Hamilton, [_Time Series Analysis_](https://www.amazon.com/Time-Analysis-James-Douglas-Hamilton/dp/0691042896) (1994)
  + `koopman_smoother`: S.J. Koopman, ["Disturbance Smoother for State Space Models"](https://www.jstor.org/stable/2336762) (_Biometrika_, 1993)
- Simulation smoothers:
  + `carter_kohn_smoother`: C.K. Carter and R. Kohn, ["On Gibbs Sampling for State Space Models"](https://www.jstor.org/stable/2337125) (_Biometrika_, 1994)
  + `durbin_koopman_smoother`: J. Durbin and S.J. Koopman, ["A Simple and Efficient Simulation Smoother for State Space Time Series Analysis"](https://www.jstor.org/stable/4140605) (_Biometrika_, 2002)

Note that not all the routines are implemented for the most general state-space
representation above. In particular, `koopman_smoother` and
`durbin_koopman_smoother` assume that `MM` is zero and will error out if a
nonzero `MM` is passed in.