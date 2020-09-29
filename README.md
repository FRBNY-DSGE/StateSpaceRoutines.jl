# State-Space Routines

[![Build Status](https://travis-ci.org/FRBNY-DSGE/StateSpaceRoutines.jl.svg)](https://travis-ci.org/FRBNY-DSGE/StateSpaceRoutines.jl)
 [![Coverage Status](https://coveralls.io/repos/github/FRBNY-DSGE/StateSpaceRoutines.jl/badge.svg?branch=master)](https://coveralls.io/github/FRBNY-DSGE/StateSpaceRoutines.jl?branch=master)
 
This package implements some common routines for state-space models. Provided algorithms include:

- Kalman filter (`kalman_filter`)
- Chandrasekhar recursions (`chand_recursion`): ["Using the "Chandrasekhar Recursions" for Likelihood Evaluation of DSGE Models"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2976646) (2012)
- Tempered particle filter (`tempered_particle_filter`): ["Tempered Particle Filtering"](https://federalreserve.gov/econresdata/feds/2016/files/2016072pap.pdf) (2016)
- Kalman smoothers:
  + `hamilton_smoother`: James Hamilton, [_Time Series Analysis_](https://www.amazon.com/Time-Analysis-James-Douglas-Hamilton/dp/0691042896) (1994)
  + `koopman_smoother`: S.J. Koopman, ["Disturbance Smoother for State Space Models"](https://www.jstor.org/stable/2336762) (_Biometrika_, 1993)
- Simulation smoothers:
  + `carter_kohn_smoother`: C.K. Carter and R. Kohn, ["On Gibbs Sampling for State Space Models"](https://www.jstor.org/stable/2337125) (_Biometrika_, 1994)
  + `durbin_koopman_smoother`: J. Durbin and S.J. Koopman, ["A Simple and Efficient Simulation Smoother for State Space Time Series Analysis"](https://www.jstor.org/stable/4140605) (_Biometrika_, 2002)

## Installation
`StateSpaceRoutines.jl` is a registered Julia package in the [`General`](https://github.com/JuliaRegistries/General) registry.  To install, open your Julia REPL, type `]` (enter package manager), and run

```julia
pkg> add StateSpaceRoutines
```

## Versioning
`StateSpaceRoutines.jl` is currently compatible with Julia `1.x`.

To use `StateSpaceRoutines.jl` with Julia `v0.7`, please check out tag `0.2.0`. To do this, click on the drop-down menu that reads `branch: master` on the left-hand side of the page. Select `tags`, then `v0.2.0`.  If you've already cloned the repo, you can simply run `git checkout v0.2.0`.

## Precompilation

The `StateSpaceRoutines.jl` package is not precompiled by default because when running code in parallel, we want to re-compile
the copy of `StateSpaceRoutines.jl` on each processor to guarantee the right version of the code is being used. If users do not
anticipate using parallelism, then users ought to change the first line of `src/StateSpaceRoutines.jl` from

```
isdefined(Base, :__precompile__) && __precompile__(false)
```

to

```
isdefined(Base, :__precompile__) && __precompile__(true)
```


## Linear Estimation

### Linear State Space System
```
s_{t+1} = C + T*s_t + R*ϵ_t    (transition equation)
y_t     = D + Z*s_t + u_t     (measurement equation)

ϵ_t ∼ N(0, Q)
u_t ∼ N(0, E)
Cov(ϵ_t, u_t) = 0
```


### Time-Invariant Methods

```
kalman_filter(y, T, R, C, Q, Z, D, E, s_0 = Vector(), P_0 = Matrix(); outputs = [:loglh, :pred, :filt], Nt0 = 0)
chand_recursion(y, T, R, C, Q, Z, D, E, s_pred = Vector(), P_pred = Matrix(); allout = false, Nt0 = 0, tol = 0.0)
tempered_particle_filter(y, Φ, Ψ, F_ϵ, F_u, s_init; verbose = :high, include_presample = true, fixed_sched = [], r_star = 2, c = 0.3, accept_rate = 0.4, target = 0.4, xtol = 0, resampling_method = :systematic, N_MH = 1, n_particles = 1000, Nt0 = 0, adaptive = true, allout = true, parallel = false)

hamilton_smoother(y, T, R, C, Q, Z, D, E, s_0, P_0; Nt0 = 0)
koopman_smoother(y, T, R, C, Q, Z, D, s_0, P_0, s_pred, P_pred; Nt0 = 0)
carter_kohn_smoother(y, T, R, C, Q, Z, D, E, s_0, P_0; Nt0 = 0, draw_states = true)
durbin_koopman_smoother(y, T, R, C, Q, Z, D, E, s_0, P_0; Nt0 = 0, draw_states = true)
```

For more information, see the docstring for each function (e.g. enter `?kalman_filter` in the REPL).

### Regime-Switching Methods

All of the provided algorithms can handle time-varying state-space systems. To do this, define `regime_indices`, a `Vector{Range{Int64}}` of length `n_regimes`, where `regime_indices[i]` indicates the range of periods `t` in regime `i`. Let `T_i`, `R_i`, etc. denote the state-space matrices in regime `i`. Then the state space is given by:

```
s_{t+1} = C_i + T_i*s_t + R_i*ϵ_t    (transition equation)
y_t     = D_i + Z_i*s_t + u_t        (measurement equation)

ϵ_t ∼ N(0, Q_i)
u_t ∼ N(0, E_i)
```

Letting `Ts = [T_1, ..., T_{n_regimes}]`, etc., we can then call the time-varying methods of the algorithms:

```
kalman_filter(regime_indices, y, Ts, Rs, Cs, Qs, Zs, Ds, Es, s_0 = Vector(), P_0 = Matrix(); outputs = [:loglh, :pred, :filt], Nt0 = 0)

hamilton_smoother(regime_indices, y, Ts, Rs, Cs, Qs, Zs, Ds, Es, s_0, P_0; Nt0 = 0)
koopman_smoother(regime_indices, y, Ts, Rs, Cs, Qs, Zs, Ds, s_0, P_0, s_pred, P_pred; Nt0 = 0)
carter_kohn_smoother(regime_indices, y, Ts, Rs, Cs, Qs, Zs, Ds, Es, s_0, P_0; Nt0 = 0, draw_states = true)
durbin_koopman_smoother(regime_indices, y, Ts, Rs, Cs, Qs, Zs, Ds, Es, s_0, P_0; Nt0 = 0, draw_states = true)
```

## Nonlinear Estimation

The tempered particle filter is a particle filtering method which can approximate the log-likelihood value implied by a general (potentially non-linear) state space system with the following representation:

### General State Space System
```
s_{t+1} = Φ(s_t, ϵ_t)        (transition equation)
y_t     = Ψ(s_t) + u_t       (measurement equation)

ϵ_t ∼ F_ϵ(∙; θ)
u_t ∼ N(0, E)
Cov(ϵ_t, u_t) = 0
```
- The documentation and code are located in [src/filters/tempered_particle_filter](https://github.com/FRBNY-DSGE/StateSpaceRoutines.jl/tree/master/src/filters/tempered_particle_filter).
- The example is located in [docs/examples/tempered_particle_filter](https://github.com/FRBNY-DSGE/StateSpaceRoutines.jl/tree/master/docs/examples/tempered_particle_filter)
- The paper proposing and analyzing the method is [Herbst and Schorfheide (2019)](https://www.sciencedirect.com/science/article/pii/S0304407618302033)

## Disclaimer
Copyright Federal Reserve Bank of New York. You may reproduce, use, modify, make derivative works of, and distribute and this code in whole or in part so long as you keep this notice in the documentation associated with any distributed works. Neither the name of the Federal Reserve Bank of New York (FRBNY) nor the names of any of the authors may be used to endorse or promote works derived from this code without prior written permission. Portions of the code attributed to third parties are subject to applicable third party licenses and rights. By your use of this code you accept this license and any applicable third party license.

THIS CODE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY WARRANTIES OR CONDITIONS OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, EXCEPT TO THE EXTENT THAT THESE DISCLAIMERS ARE HELD TO BE LEGALLY INVALID. FRBNY IS NOT, UNDER ANY CIRCUMSTANCES, LIABLE TO YOU FOR DAMAGES OF ANY KIND ARISING OUT OF OR IN CONNECTION WITH USE OF OR INABILITY TO USE THE CODE, INCLUDING, BUT NOT LIMITED TO DIRECT, INDIRECT, INCIDENTAL, CONSEQUENTIAL, PUNITIVE, SPECIAL OR EXEMPLARY DAMAGES, WHETHER BASED ON BREACH OF CONTRACT, BREACH OF WARRANTY, TORT OR OTHER LEGAL OR EQUITABLE THEORY, EVEN IF FRBNY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES OR LOSS AND REGARDLESS OF WHETHER SUCH DAMAGES OR LOSS IS FORESEEABLE.
