"""
```
durbin_koopman_smoother(data, TTT, RRR, CCC, QQ, ZZ, DD, z0, P0;
    n_presample_periods = 0, draw_states = true)

durbin_koopman_smoother(regime_indices, data, TTTs, RRRs, CCCs,
    QQs, ZZs, DDs, z0, P0; n_presample_periods = 0, draw_states = true)
```

This program is a simulation smoother based on Durbin and Koopman's
\"A Simple and Efficient Simulation Smoother for State Space Time Series
Analysis\" (Biometrika, 2002).

Unlike other simulation smoothers (for example, that of Carter and Kohn,
1994), this method does not require separate draws for each period, draws
of the state vectors, or even draws from a conditional distribution.
Instead, vectors of shocks are drawn from the unconditional distribution
of shocks, which is then corrected (via a Kalman smoothing step), to
yield a draw of shocks conditional on the data. This is then used to
generate a draw of states conditional on the data. Drawing the states in
this way is much more efficient than other methods, as it avoids the need
for multiple draws of state vectors (requiring singular value
decompositions), as well as inverting state covariance matrices
(requiring the use of the computationally intensive and relatively
erratic Moore-Penrose pseudoinverse).

The state space is given by:

```
z_{t+1} = CCC + TTT*z_t + RRR*ϵ_t          (transition equation)
y_t     = DD  + ZZ*z_t  + MM*ϵ_t  + η_t    (measurement equation)

ϵ_t ∼ N(0, QQ)
η_t ∼ N(0, EE)
```

### Inputs

- `data`: `Ny` x `T` matrix containing data `y(1), ... , y(T)`
- `z0`: `Nz` x 1 initial state vector
- `P0`: `Nz` x `Nz` initial state covariance matrix
- `pred`: `Nz` x `T` matrix of one-step predicted state vectors `z_{t|t-1}`
  (from the Kalman filter)
- `vpred`: `Nz` x `Nz` x `T` array of mean squared errors `P_{t|t-1}` of
  predicted state vectors

**Method 1 only:** state-space system matrices `TTT`, `RRR`, `CCC`, `QQ`, `ZZ`,
`DD`. See `?kalman_filter`

**Method 2 only:** `regime_indices` and system matrices for each regime `TTTs`,
`RRRs`, `CCCs`, `QQs`, `ZZs`, `DDs`. See `?kalman_filter`

where:

- `T`: number of periods for which we have data
- `Nz`: number of states
- `Ne`: number of shocks
- `Ny`: number of observables

### Keyword Arguments

- `n_presample_periods`: if greater than 0, the returned smoothed states and
  shocks matrices will be shorter by that number of columns (taken from the
  beginning)
- `draw_states`: indicates whether to draw smoothed states from the distribution
  `N(z_{t|T}, P_{t|T})` or to use the mean `z_{t|T}` (reducing to the Koopman
  smoother). Defaults to `true`

### Outputs

- `smoothed_states`: `Nz` x `T` matrix of smoothed states `z_{t|T}`
- `smoothed_shocks`: `Ne` x `T` matrix of smoothed shocks `ϵ_{t|T}`
"""
function durbin_koopman_smoother{S<:AbstractFloat}(data::Matrix{S},
    TTT::Matrix{S}, RRR::Matrix{S}, CCC::Vector{S},
    QQ::Matrix{S}, ZZ::Matrix{S}, DD::Vector{S},
    MM::Matrix{S}, EE::Matrix{S}, z0::Vector{S}, P0::Matrix{S};
    n_presample_periods::Int = 0, draw_states::Bool = true)

    T = size(data, 2)
    regime_indices = Range{Int64}[1:T]

    durbin_koopman_smoother(regime_indices, data, Matrix{S}[TTT], Matrix{S}[RRR], Vector{S}[CCC],
        Matrix{S}[QQ], Matrix{S}[ZZ], Vector{S}[DD], Matrix{S}[MM], Matrix{S}[EE], z0, P0;
        n_presample_periods = n_presample_periods, draw_states = draw_states)
end

function durbin_koopman_smoother{S<:AbstractFloat}(regime_indices::Vector{Range{Int64}},
    data::Matrix{S}, TTTs::Vector{Matrix{S}}, RRRs::Vector{Matrix{S}}, CCCs::Vector{Vector{S}},
    QQs::Vector{Matrix{S}}, ZZs::Vector{Matrix{S}}, DDs::Vector{Vector{S}},
    MMs::Vector{Matrix{S}}, EEs::Vector{Matrix{S}}, z0::Vector{S}, P0::Matrix{S};
    n_presample_periods::Int = 0, draw_states::Bool = true)

    n_regimes = length(regime_indices)

    # Dimensions
    T  = size(data,    2) # number of periods of data
    Nz = size(TTTs[1], 1) # number of states
    Ne = size(RRRs[1], 2) # number of shocks
    Ny = size(ZZs[1],  1) # number of observables

    # Draw initial state α_0+ and sequence of shocks η+
    if draw_states
        U, eig, _ = svd(P0)
        α_plus_t  = U * diagm(sqrt(eig)) * randn(Nz)
        η_plus    = sqrt(QQs[1]) * randn(Ne, T)
    else
        α_plus_t  = zeros(S, Nz)
        η_plus    = zeros(S, Ne, T)
    end

    # Produce "fake" states and observables (α+ and y+) by
    # iterating the state-space system forward
    α_plus       = zeros(S, Nz, T)
    y_plus       = zeros(S, Ny, T)

    for i = 1:n_regimes
        regime_periods = regime_indices[i]

        # Get state-space system matrices for this regime
        TTT, RRR, CCC = TTTs[i], RRRs[i], CCCs[i]
        QQ,  ZZ,  DD  = QQs[i],  ZZs[i],  DDs[i]

        for t in regime_periods
            η_plus_t = η_plus[:, t]
            α_plus_t = TTT*α_plus_t + RRR*η_plus_t + CCC

            α_plus[:, t] = α_plus_t
            y_plus[:, t] = ZZ*α_plus_t + DD
        end
    end

    # Replace fake data with NaNs wherever actual data has NaNs
    y_plus[isnan(data)] = NaN

    # Compute y* = y - y+
    y_star = data - y_plus

    # Run the Kalman filter
    # Note that we pass in `zeros(size(D))` instead of `D` because the
    # measurement equation for `data_star` has no constant term
    _, pred, vpred, _ = kalman_filter(regime_indices, y_star, TTTs, RRRs, CCCs,
                            QQs, ZZs, fill(zeros(Ny), n_regimes),
                            MMs, EEs, z0, P0)

    # Kalman smooth
    α_hat_star, η_hat_star = koopman_smoother(regime_indices, y_star, TTTs, RRRs, CCCs,
                                 QQs, ZZs, fill(zeros(Ny), n_regimes), MMs, EEs,
                                 z0, P0, pred, vpred)

    # Compute draw (states and shocks)
    smoothed_states = α_plus + α_hat_star
    smoothed_shocks = η_plus + η_hat_star

    # Trim the presample if needed
    if n_presample_periods > 0
        mainsample_periods = n_presample_periods+1:T

        smoothed_states = smoothed_states[:, mainsample_periods]
        smoothed_shocks = smoothed_shocks[:, mainsample_periods]
    end

    return smoothed_states, smoothed_shocks
end