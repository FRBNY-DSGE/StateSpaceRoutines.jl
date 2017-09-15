# State-Space Routines

[![Build Status](https://travis-ci.org/FRBNY-DSGE/DSGE.jl.svg)](https://travis-ci.org/FRBNY-DSGE/StateSpaceRoutines.jl)

This package implements some common routines for state-space models with the
following representation:

```
z_{t+1} = CCC + TTT*z_t + RRR*ϵ_t    (transition equation)
y_t     = DD  + ZZ*z_t  + η_t        (measurement equation)

ϵ_t ∼ N(0, QQ)
η_t ∼ N(0, EE)
Cov(ϵ_t, η_t) = 0
```

The provided algorithms are:

- Kalman filter (`kalman_filter`)
- Tempered particle filter (`tempered_particle_filter`): ["Tempered Particle Filtering"](https://federalreserve.gov/econresdata/feds/2016/files/2016072pap.pdf) (2016)
- Kalman smoothers:
  + `hamilton_smoother`: James Hamilton, [_Time Series Analysis_](https://www.amazon.com/Time-Analysis-James-Douglas-Hamilton/dp/0691042896) (1994)
  + `koopman_smoother`: S.J. Koopman, ["Disturbance Smoother for State Space Models"](https://www.jstor.org/stable/2336762) (_Biometrika_, 1993)
- Simulation smoothers:
  + `carter_kohn_smoother`: C.K. Carter and R. Kohn, ["On Gibbs Sampling for State Space Models"](https://www.jstor.org/stable/2337125) (_Biometrika_, 1994)
  + `durbin_koopman_smoother`: J. Durbin and S.J. Koopman, ["A Simple and Efficient Simulation Smoother for State Space Time Series Analysis"](https://www.jstor.org/stable/4140605) (_Biometrika_, 2002)


## Time-Invariant Methods

```
kalman_filter(data, TTT, RRR, CCC, QQ, ZZ, DD, EE, z0 = Vector(), P0 = Matrix(); allout = true, n_presample_periods = 0)
tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; verbose, fixed_sched, r_star, c, accept_rate, target, xtol,
resampling_method, N_MH, n_particles, n_presample_periods, adaptive, allout, parallel)

hamilton_smoother(data, TTT, RRR, CCC, QQ, ZZ, DD, EE, z0, P0; n_presample_periods = 0)
koopman_smoother(data, TTT, RRR, CCC, QQ, ZZ, DD, z0, P0, pred, vpred; n_presample_periods = 0)
carter_kohn_smoother(data, TTT, RRR, CCC, QQ, ZZ, DD, EE, z0, P0; n_presample_periods = 0, draw_states = true)
durbin_koopman_smoother(data, TTT, RRR, CCC, QQ, ZZ, DD, EE, z0, P0; n_presample_periods = 0, draw_states = true)
```

For more information, see the documentation for each function (e.g. by entering
`?kalman_filter` in the REPL).







## Regime-Switching Methods

All of the provided algorithms can handle time-varying state-space systems. To
do this, define `regime_indices`, a `Vector{Range{Int64}}` of length
`n_regimes`, where `regime_indices[i]` indicates the range of periods `t` in
regime `i`. Let `TTT_i`, `RRR_i`, etc. denote the state-space matrices in regime
`i`. Then the state space is given by:

```
z_{t+1} = CCC_i + TTT_i*z_t + RRR_i*ϵ_t    (transition equation)
y_t     = DD_i  + ZZ_i*z_t  + η_t          (measurement equation)

ϵ_t ∼ N(0, QQ_i)
η_t ∼ N(0, EE_i)
```

Letting `TTTs = [TTT_1, ..., TTT_{n_regimes}]`, etc., we can then call the time-
varying methods of the algorithms:

```
kalman_filter(regime_indices, data, TTTs, RRRs, CCCs, QQs, ZZs, DDs, EEs, z0 = Vector(), P0 = Matrix(); allout = true, n_presample_periods = 0)

hamilton_smoother(regime_indices, data, TTTs, RRRs, CCCs, QQs, ZZs, DDs, EEs, z0, P0; n_presample_periods = 0)
koopman_smoother(regime_indices, data, TTTs, RRRs, CCCs, QQs, ZZs, DDs, z0, P0, pred, vpred; n_presample_periods = 0)
carter_kohn_smoother(regime_indices, data, TTTs, RRRs, CCCs, QQs, ZZs, DDs, EEs, z0, P0; n_presample_periods = 0, draw_states = true)
durbin_koopman_smoother(regime_indices, data, TTTs, RRRs, CCCs, QQs, ZZs, DDs, EEs, z0, P0; n_presample_periods = 0, draw_states = true)
```





