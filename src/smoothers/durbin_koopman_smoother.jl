"""
```
durbin_koopman_smoother{S<:AbstractFloat}(m::AbstractModel,
    df::DataFrame, system::System, z0::Vector{S}, P0::Matrix{S};
    cond_type::Symbol = :none, include_presample::Bool = false)

durbin_koopman_smoother{S<:AbstractFloat}(m::AbstractModel,
    data:Matrix{S}, system::System, z0::Vector{S}, P0::Matrix{S};
    include_presample::Bool = false)

durbin_koopman_smoother{S<:AbstractFloat}(m::AbstractModel,
    df::DataFrame, T::Matrix{S}, R::Matrix{S}, C::Array{S}, Q::Matrix{S},
    Z::Matrix{S}, D::Matrix{S}, z0::Array{S}, P0::Matrix{S};
    cond_type::Symbol = :none, include_presample::Bool = false)

durbin_koopman_smoother{S<:AbstractFloat}(m::AbstractModel,
    data::Matrix{S}, T::Matrix{S}, R::Matrix{S}, C::Array{S}, Q::Matrix{S},
    Z::Matrix{S}, D::Matrix{S}, z0::Array{S}, P0::Matrix{S};
    include_presample::Bool = false)
```
This program is a simulation smoother based on Durbin and Koopman's
\"A Simple and Efficient Simulation Smoother for State Space Time Series
Analysis\" (Biometrika, 2002). The algorithm has been simplified for the
case in which there is no measurement error, and the model matrices do
not vary with time.

Unlike other simulation smoothers (for example, that of Carter and Kohn,
1994), this method does not require separate draws for each period, draws
of the state vectors, or even draws from a conditional distribution.
Instead, vectors of shocks are drawn from the unconditional distribution
of shocks, which is then corrected (via a Kalman Smoothing step), to
yield a draw of shocks conditional on the data. This is then used to
generate a draw of states conditional on the data. Drawing the states in
this way is much more efficient than other methods, as it avoids the need
for multiple draws of state vectors (requiring singular value
decompositions), as well as inverting state covariance matrices
(requiring the use of the computationally intensive and relatively
erratic Moore-Penrose pseudoinverse).

### Inputs:

- `m`: model object
- `data`: the (`Ny` x `Nt`) matrix of observable data
- `T`: the (`Nz` x `Nz`) transition matrix
- `R`: the (`Nz` x `Ne`) matrix translating shocks to states
- `C`: the (`Nz` x 1) constant vector in the transition equation
- `Q`: the (`Ne` x `Ne`) covariance matrix for the shocks
- `Z`: the (`Ny` x `Nz`) measurement matrix
- `D`: the (`Ny` x 1) constant vector in the measurement equation
- `z0`: the (`Nz` x 1) initial (time 0) states vector
- `P0`: the (`Nz` x `Nz`) initial (time 0) state covariance matrix. If
- `cond_type`: optional keyword argument specifying the conditional data type:
  one of `:none`, `:semi`, or `:full`. This is only necessary when a DataFrame
  (as opposed to a data matrix) is passed in, so that `df_to_matrix` knows how
  many periods of data to keep
- `include_presample`: indicates whether or not to return presample periods in
  the returned smoothed states and shocks. Defaults to `false`

Where:

- `Nz`: number of states
- `Ny`: number of observables
- `Ne`: number of shocks
- `Nt`: number of periods for which we have data

### Outputs:

- `α_hat`: the (`Nz` x `Nt`) matrix of smoothed states.
- `η_hat`: the (`Ne` x `Nt`) matrix of smoothed shocks.

If `n_presample_periods(m)` is nonzero, the `α_hat` and `η_hat` matrices will be
shorter by that number of columns (taken from the beginning).

### Notes

The state space model is defined as follows:
```
y(t) = Z*α(t) + D             (state or transition equation)
α(t+1) = T*α(t) + R*η(t+1)    (measurement or observation equation)
```
"""
function durbin_koopman_smoother{S<:AbstractFloat}(data::Matrix{S},
    TTT::Matrix{S}, RRR::Matrix{S}, CCC::Vector{S},
    QQ::Matrix{S}, ZZ::Matrix{S}, DD::Vector{S},
    MM::Matrix{S}, EE::Matrix{S}, z0::Vector{S}, P0::Matrix{S};
    n_presample_periods::Int = 0, draw_states::Bool = true)

    T = size(data, 2)
    regime_indices = Range{Int64}[1:T]

    durbin_koopman_smoother(regime_indices, data, Matrix{S}[TTT], Matrix{S}[RRR], Vector{S}[CCC],
        Matrix{S}[QQ], Matrix{S}[ZZ], Vector{S}[DD], Vector{S}[MM], Vector{S}[EE], z0, P0;
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
        η_plus    = sqrt(QQs[1]) * randn(Ne, Nt)
    else
        α_plus_t  = zeros(S, Nz)
        η_plus    = zeros(S, Ne, T)
    end

    # Produce "fake" states and observables (α+ and y+) by
    # iterating the state-space system forward
    α_plus       = zeros(S, Nz, T)
    y_plus       = zeros(S, Ny, T)

    for i = 1:n_regimes
        # Get state-space system matrices for this regime
        regime_periods = regime_indices[i]

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
                            QQs, ZZs, fill(zeros(Ny), n_regimes), MMs, EEs,
                            z0, P0)

    # Kalman smooth
    α_hat_star, η_hat_star = koopman_smoother(regime_indices, y_star, TTTs, RRRs, CCCs,
                                 QQs, ZZs, fill(zeros(Ny), n_regimes),
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