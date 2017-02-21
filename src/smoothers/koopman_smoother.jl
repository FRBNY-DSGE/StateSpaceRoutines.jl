"""
```
koopman_smoother{S<:AbstractFloat}(m::AbstractModel, df::DataFrame,
    system::System, z0::Vector{S}, P0::Matrix{S}, pred::Matrix{S}, vpred::Array{S, 3};
    cond_type::Symbol = :none, include_presample::Bool = false)

koopman_smoother{S<:AbstractFloat}(m::AbstractModel, data::Matrix{S},
    system::System, z0::Vector{S}, P0::Matrix{S}, pred::Matrix{S}, vpred::Array{S, 3};
    include_presample::Bool = false)

koopman_smoother{S<:AbstractFloat}(m::AbstractModel, df::DataFrame,
    T::Matrix{S}, R::Matrix{S}, C::Array{S}, Q::Matrix{S}, Z::Matrix{S},
    D::Vector{S}, z0::Vector{S}, P0::Matrix{S}, pred::Matrix{S}, vpred::Array{S, 3};
    cond_type::Symbol = :none, include_presample::Bool = false)

koopman_smoother{S<:AbstractFloat}(m::AbstractModel, data::Matrix{S}
    T::Matrix{S}, R::Matrix{S}, C::Array{S}, Q::Matrix{S}, Z::Matrix{S},
    D::Vector{S}, z0::Vector{S}, P0::Matrix{S}, pred::Matrix{S}, vpred::Array{S, 3};
    include_presample::Bool = false)
```
This is a Kalman Smoothing program based on S.J. Koopman's \"Disturbance
Smoother for State Space Models\" (Biometrika, 1993), as specified in
Durbin and Koopman's \"A Simple and Efficient Simulation Smoother for
State Space Time Series Analysis\" (Biometrika, 2002). The algorithm has been
simplified for the case in which there is no measurement error, and the
model matrices do not vary with time.

Unlike other Kalman Smoothing programs, there is no need to invert
singular matrices using the Moore-Penrose pseudoinverse (pinv), which
should lead to efficiency gains and fewer inversion problems. Also, the
states vector and the corresponding matrices do not need to be augmented
to include the shock innovations. Instead they are saved automatically
in the `eta_hat` matrix.

### Inputs:

- `m`: model object
- `data`: the (`Ny` x `T`) matrix of observable data
- `T`: the (`Nz` x `Nz`) transition matrix
- `R`: the (`Nz` x `Ne`) matrix translating shocks to states
- `C`: the (`Nz` x 1) constant vector in the transition equation
- `Q`: the (`Ne` x `Ne`) covariance matrix for the shocks
- `Z`: the (`Ny` x `Nz`) measurement matrix
- `D`: the (`Ny` x 1) constant vector in the measurement equation
- `z0`: the (`Nz` x 1) initial (time 0) states vector
- `P0`: the (`Nz` x `Nz`) initial (time 0) state covariance matrix
- `pred`: the (`Nz` x `T`) matrix of one-step-ahead predicted states (from the
  Kalman Filter)
- `vpred`: the (`Nz` x `Nz` x `T`) matrix of one-step-ahead predicted
  covariance matrices
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
- `T`: number of periods for which we have data

### Outputs:

- `α_hat`: the (`Nz` x `T`) matrix of smoothed states
- `η_hat`: the (`Ne` x `T`) matrix of smoothed shocks

If `n_presample_periods(m)` is nonzero, the `α_hat` and `η_hat` matrices will be
shorter by that number of columns (taken from the beginning).

### Notes

The state space model is defined as follows:
```
y(t) = Z*α(t) + D             (state or transition equation)
α(t+1) = T*α(t) + R*η(t+1)    (measurement or observation equation)
```
"""
function koopman_smoother{S<:AbstractFloat}(data::Matrix{S},
    TTT::Matrix{S}, RRR::Matrix{S}, CCC::Vector{S},
    QQ::Matrix{S}, ZZ::Matrix{S}, DD::Vector{S},
    z0::Vector{S}, P0::Matrix{S}, pred::Matrix{S}, vpred::Array{S, 3};
    n_presample_periods::Int = 0)

    T = size(data, 2)
    regime_indices = Range{Int64}[1:T]

    koopman_smoother(regime_indices, data, Matrix{S}[TTT], Matrix{S}[RRR], Vector{S}[CCC],
        Matrix{S}[QQ], Matrix{S}[ZZ], Vector{S}[DD], z0, P0, pred, vpred;
        n_presample_periods = n_presample_periods)
end

function koopman_smoother{S<:AbstractFloat}(regime_indices::Vector{Range{Int64}},
    data::Matrix{S}, TTTs::Vector{Matrix{S}}, RRRs::Vector{Matrix{S}}, CCCs::Vector{Vector{S}},
    QQs::Vector{Matrix{S}}, ZZs::Vector{Matrix{S}}, DDs::Vector{Vector{S}},
    z0::Vector{S}, P0::Matrix{S}, pred::Matrix{S}, vpred::Array{S, 3};
    n_presample_periods::Int = 0)

    n_regimes = length(regime_indices)

    # Dimensions
    T  = size(data,    2) # number of periods of data
    Nz = size(TTTs[1], 1) # number of states

    # Call disturbance smoother
    smoothed_disturbances, smoothed_shocks = koopman_disturbance_smoother(regime_indices, data,
                                                 TTTs, RRRs, QQs, ZZs, DDs, pred, vpred)

    # Initialize outputs
    smoothed_states = zeros(S, Nz, T)

    r     = smoothed_disturbances[:, 1]
    α_hat = z0 + P0*r
    smoothed_states[:, 1] = α_hat

    for i = 1:n_regimes
        # Get state-space system matrices for this regime
        regime_periods = regime_indices[i]

        TTT, RRR     = TTTs[i], RRRs[i]
        QQ,  ZZ,  DD = QQs[i],  ZZs[i],  DDs[i]

        for t in regime_periods
            # t = 1 has already been initialized
            t == 1 ? continue : nothing

            r     = smoothed_disturbances[:, t]
            α_hat = TTT*α_hat + RRR*QQ*RRR'*r

            smoothed_states[:, t] = α_hat
        end
    end

    # Trim the presample if needed
    if n_presample_periods > 0
        mainsample_periods = n_presample_periods+1:T

        smoothed_states = smoothed_states[:, mainsample_periods]
        smoothed_shocks = smoothed_shocks[:, mainsample_periods]
    end

    return smoothed_states, smoothed_shocks
end

"""
```
koopman_disturbance_smoother{S<:AbstractFloat}(m::AbstractModel,
    data::Matrix{S}, T::Matrix{S}, R::Matrix{S}, C::Array{S}, Q::Matrix{S},
    Z::Matrix{S}, D::Array{S}, pred::Matrix{S}, vpred::Array{S, 3})
```

This is a Kalman Smoothing program based on S.J. Koopman's \"Disturbance
Smoother for State Space Models\" (Biometrika, 1993), as specified in
Durbin and Koopman's \"A Simple and Efficient Simulation Smoother for
State Space Time Series Analysis\" (Biometrika, 2002). The algorithm has been
simplified for the case in which there is no measurement error, and the
model matrices do not vary with time.

This disturbance smoother is intended for use with the state smoother
`koopman_smoother` from the same papers (Koopman 1993, Durbin and Koopman
2002). It produces a matrix of vectors, `r`, that is used for state
smoothing, and an optional matrix, `eta_hat`, containing the smoothed
shocks. It has been adjusted to account for the possibility of missing
values in the data, and to accommodate the zero bound model, which
requires that no anticipated shocks occur before the zero bound window,
which is achieved by setting the entries in the `Q` matrix corresponding to
the anticipated shocks to zero in those periods.

### Inputs:

- `m`: model object
- `data`: the (`Ny` x `T`) matrix of observable data
- `T`: the (`Nz` x `Nz`) transition matrix
- `R`: the (`Nz` x `Ne`) matrix translating shocks to states
- `C`: the (`Nz` x 1) constant vector in the transition equation
- `Q`: the (`Ne` x `Ne`) covariance matrix for the shocks
- `Z`: the (`Ny` x `Nz`) measurement matrix
- `D`: the (`Ny` x 1) constant vector in the measurement equation
- `pred`: the (`Nz` x `T`) matrix of one-step-ahead predicted states (from the
  Kalman Filter)
- `vpred`: the (`Nz` x `Nz` x `T`) matrix of one-step-ahead predicted
  covariance matrices

Where:

- `Nz`: number of states
- `Ny`: number of observables
- `Ne`: number of shocks
- `T`: number of periods for which we have data

### Outputs:

- `r`: the (`Nz` x `T`) matrix used for state smoothing
- `η_hat`: the (`Ne` x `T`) matrix of smoothed shocks

### Notes

The state space model is defined as follows:
```
y(t) = Z*α(t) + D             (state or transition equation)
α(t+1) = T*α(t) + R*η(t+1)    (measurement or observation equation)
```
"""
function koopman_disturbance_smoother{S<:AbstractFloat}(data::Matrix{S},
    TTT::Matrix{S}, RRR::Matrix{S},
    QQ::Matrix{S}, ZZ::Matrix{S}, DD::Vector{S},
    pred::Matrix{S}, vpred::Array{S, 3},
    n_presample_periods::Int = 0)

    T = size(data, 2)
    regime_indices = Range{Int64}[1:T]

    koopman_disturbance_smoother(regime_indices, data, Matrix{S}[TTT], Matrix{S}[RRR],
        Matrix{S}[QQ], Matrix{S}[ZZ], Vector{S}[DD], pred, vpred;
        n_presample_periods = n_presample_periods)
end

function koopman_disturbance_smoother{S<:AbstractFloat}(regime_indices::Vector{Range{Int64}},
    data::Matrix{S}, TTTs::Vector{Matrix{S}}, RRRs::Vector{Matrix{S}},
    QQs::Vector{Matrix{S}}, ZZs::Vector{Matrix{S}}, DDs::Vector{Vector{S}},
    pred::Matrix{S}, vpred::Array{S, 3}; n_presample_periods::Int = 0)

    n_regimes = length(regime_indices)

    # Dimensions
    T  = size(data,    2) # number of periods of data
    Nz = size(TTTs[1], 1) # number of states
    Ne = size(RRRs[1], 2) # number of shocks

    # Initialize outputs
    smoothed_disturbances = zeros(S, Nz, T)
    smoothed_shocks       = zeros(S, Ne, T)

    r = zeros(S, Nz)

    for i = n_regimes:-1:1
        # Get state-space system matrices for this regime
        regime_periods = regime_indices[i]

        TTT, RRR     = TTTs[i], RRRs[i]
        QQ,  ZZ,  DD = QQs[i],  ZZs[i],  DDs[i]

        for t in reverse(regime_periods)
            # If an element of the vector y_t is missing (NaN) for the observation t, the
            # corresponding row is ditched from the measurement equation
            nonmissing = !isnan(data[:, t])
            y_t  = data[nonmissing, t]
            ZZ_t = ZZ[nonmissing, :]
            DD_t = DD[nonmissing]

            a = pred[:, t]
            P = vpred[:, :, t]

            F = ZZ_t*P*ZZ_t'
            v = y_t - ZZ_t*a - DD_t
            K = TTT*P*ZZ_t'/F
            L = TTT - K*ZZ_t

            r = ZZ_t'/F*v + L'*r
            smoothed_disturbances[:, t] = r

            smoothed_shocks[:, t] = QQ*RRR'*r

        end # of loop backward through this regime's periods

    end # of loop backward through regimes

    # Trim the presample if needed
    if n_presample_periods > 0
        mainsample_periods = n_presample_periods+1:T

        smoothed_disturbances = smoothed_disturbances[:, mainsample_periods]
        smoothed_shocks       = smoothed_shocks[:,       mainsample_periods]
    end

    return smoothed_disturbances, smoothed_shocks
end