"""
```
koopman_smoother(data, TTT, RRR, CCC, QQ, ZZ, DD, z0, P0, pred, vpred;
    n_presample_periods = 0)

koopman_smoother(regime_indices, data, TTTs, RRRs, CCCs, QQs, ZZs, DDs,
    z0, P0, pred, vpred; n_presample_periods = 0)
```

This is a Kalman smoothing program based on S.J. Koopman's \"Disturbance
Smoother for State Space Models\" (Biometrika, 1993), as specified in
Durbin and Koopman's \"A Simple and Efficient Simulation Smoother for
State Space Time Series Analysis\" (Biometrika, 2002).

Unlike other Kalman smoothing programs, there is no need to invert
singular matrices using the Moore-Penrose pseudoinverse (`pinv`), which
should lead to efficiency gains and fewer inversion problems. Also, the
states vector and the corresponding matrices do not need to be augmented
to include the shock innovations. Instead they are saved automatically
in the `smoothed_shocks` matrix.

The state space is given by:

```
z_{t+1} = CCC + TTT*z_t + RRR*ϵ_t    (transition equation)
y_t     = DD  + ZZ*z_t  + η_t        (measurement equation)

ϵ_t ∼ N(0, QQ)
η_t ∼ N(0, EE)
Cov(ϵ_t, η_t) = 0
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

### Outputs

- `smoothed_states`: `Nz` x `T` matrix of smoothed states `z_{t|T}`
- `smoothed_shocks`: `Ne` x `T` matrix of smoothed shocks `ϵ_{t|T}`
"""
function koopman_smoother{S<:AbstractFloat}(data::Matrix{S},
    TTT::Matrix{S}, RRR::Matrix{S}, CCC::Vector{S},
    QQ::Matrix{S}, ZZ::Matrix{S}, DD::Vector{S}, EE::Matrix{S},
    z0::Vector{S}, P0::Matrix{S}, pred::Matrix{S}, vpred::Array{S, 3};
    n_presample_periods::Int = 0)

    T = size(data, 2)
    regime_indices = Range{Int64}[1:T]

    koopman_smoother(regime_indices, data, Matrix{S}[TTT], Matrix{S}[RRR], Vector{S}[CCC],
        Matrix{S}[QQ], Matrix{S}[ZZ], Vector{S}[DD], Matrix{S}[EE],
        z0, P0, pred, vpred;
        n_presample_periods = n_presample_periods)
end

function koopman_smoother{S<:AbstractFloat}(regime_indices::Vector{Range{Int64}},
    data::Matrix{S}, TTTs::Vector{Matrix{S}}, RRRs::Vector{Matrix{S}}, CCCs::Vector{Vector{S}},
    QQs::Vector{Matrix{S}}, ZZs::Vector{Matrix{S}}, DDs::Vector{Vector{S}}, EEs::Vector{Matrix{S}},
    z0::Vector{S}, P0::Matrix{S}, pred::Matrix{S}, vpred::Array{S, 3};
    n_presample_periods::Int = 0)

    # Dimensions
    n_regimes = length(regime_indices)
    T  = size(data,    2) # number of periods of data
    Nz = size(TTTs[1], 1) # number of states
    Ne = size(RRRs[1], 2) # number of shocks

    # Call disturbance smoother
    state_disturbances, observable_disturbances =
        koopman_disturbance_smoother(regime_indices, data, TTTs, RRRs,
            QQs, ZZs, DDs, EEs, pred, vpred)

    # Initialize outputs
    smoothed_states = zeros(S, Nz, T)
    smoothed_shocks = zeros(S, Ne, T)

    r     = state_disturbances[:, 1]
    α_hat = zeros(S, Nz) # initialize dummy value s.t. α_hat is in scope

    for i = 1:n_regimes
        regime_periods = regime_indices[i]

        # Get state-space system matrices for this regime
        TTT, RRR     = TTTs[i], RRRs[i]
        QQ,  ZZ,  DD = QQs[i],  ZZs[i],  DDs[i]

        for t in regime_periods
            r     = state_disturbances[:, t]
            α_hat = if t == 1
                z0 + P0*r
            else
                TTT*α_hat + RRR*QQ*RRR'*r
            end

            smoothed_states[:, t] = α_hat
            smoothed_shocks[:, t] = QQ*RRR'*r
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
koopman_disturbance_smoother(data, TTT, RRR, QQ, ZZ, DD, EE, pred, vpred;
    n_presample_periods = 0)

koopman_smoother(regime_indices, data, TTTs, RRRs, QQs, ZZs, DDs, EEs,
    pred, vpred; n_presample_periods = 0)
```

This disturbance smoother is intended for use with the state smoother
`koopman_smoother` from S.J. Koopman's \"Disturbance Smoother for State Space
Models\" (Biometrika, 1993), as specified in Durbin and Koopman's \"A Simple and
Efficient Simulation Smoother for State Space Time Series Analysis\"
(Biometrika, 2002).

The state space is given by:

```
z_{t+1} = CCC + TTT*z_t + RRR*ϵ_t    (transition equation)
y_t     = DD  + ZZ*z_t  + η_t        (measurement equation)

ϵ_t ∼ N(0, QQ)
η_t ∼ N(0, EE)
Cov(ϵ_t, η_t) = 0
```

### Inputs

- `data`: `Ny` x `T` matrix containing data `y(1), ... , y(T)`
- `pred`: `Nz` x `T` matrix of one-step predicted state vectors `z_{t|t-1}`
  (from the Kalman filter)
- `vpred`: `Nz` x `Nz` x `T` array of mean squared errors `P_{t|t-1}` of
  predicted state vectors

**Method 1 only:** state-space system matrices `TTT`, `RRR`, `QQ`, `ZZ`,
`DD`. See `?kalman_filter`

**Method 2 only:** `regime_indices` and system matrices for each regime `TTTs`,
`RRRs`, `QQs`, `ZZs`, `DDs`. See `?kalman_filter`

where:

- `T`: number of periods for which we have data
- `Nz`: number of states
- `Ne`: number of shocks
- `Ny`: number of observables

### Keyword Arguments

- `n_presample_periods`: if greater than 0, the returned smoothed disturbances
  and shocks matrices will be shorter by that number of columns (taken from the
  beginning)

### Outputs

- `state_disturbances`: `Nz` x `T` matrix of transition equation disturbances
  `r_t`
- `observable_disturbances`: `Ny` x `T` matrix of measurement equation
  disturbances `e_{t|T}`
"""
function koopman_disturbance_smoother{S<:AbstractFloat}(data::Matrix{S},
    TTT::Matrix{S}, RRR::Matrix{S}, QQ::Matrix{S},
    ZZ::Matrix{S}, DD::Vector{S}, EE::Matrix{S},
    pred::Matrix{S}, vpred::Array{S, 3};
    n_presample_periods::Int = 0)

    T = size(data, 2)
    regime_indices = Range{Int64}[1:T]

    koopman_disturbance_smoother(regime_indices, data, Matrix{S}[TTT], Matrix{S}[RRR],
        Matrix{S}[QQ], Matrix{S}[ZZ], Vector{S}[DD], Matrix{S}[EE],
        pred, vpred; n_presample_periods = n_presample_periods)
end

function koopman_disturbance_smoother{S<:AbstractFloat}(regime_indices::Vector{Range{Int64}},
    data::Matrix{S}, TTTs::Vector{Matrix{S}}, RRRs::Vector{Matrix{S}},
    QQs::Vector{Matrix{S}}, ZZs::Vector{Matrix{S}}, DDs::Vector{Vector{S}}, EEs::Vector{Matrix{S}},
    pred::Matrix{S}, vpred::Array{S, 3}; n_presample_periods::Int = 0)

    # Dimensions
    n_regimes = length(regime_indices)
    T  = size(data,    2) # number of periods of data
    Nz = size(TTTs[1], 1) # number of states
    Ne = size(RRRs[1], 2) # number of shocks
    Ny = size(ZZs[1],  1) # number of observables

    # Initialize outputs
    state_disturbances      = zeros(S, Nz, T)
    observable_disturbances = zeros(S, Ny, T)

    r = zeros(S, Nz)

    for i = n_regimes:-1:1
        # Get state-space system matrices for this regime
        regime_periods = regime_indices[i]

        TTT, RRR     = TTTs[i], RRRs[i]
        ZZ,  DD      = ZZs[i],  DDs[i]
        QQ,  EE      = QQs[i],  EEs[i]

        for t in reverse(regime_periods)
            # If an element of the vector y_t is missing (NaN) for the observation t, the
            # corresponding row is ditched from the measurement equation
            nonmissing = .!isnan.(data[:, t])
            y_t  = data[nonmissing, t]
            ZZ_t = ZZ[nonmissing, :]
            DD_t = DD[nonmissing]
            EE_t = EE[nonmissing, nonmissing]

            z = pred[:, t]                    # z_{t|t-1}
            P = vpred[:, :, t]                # P_{t|t-1} = Var s_{t|t-1}
            V = ZZ_t*P*ZZ_t' + EE_t           # V_{t|t-1} = Var y_{t|t-1} = ZZ*P_{t|t-1}*ZZ' + EE
            dy = y_t - ZZ_t*z - DD_t          # dy = y_t - y_{t|t-1} = prediction error

            K = TTT*P*ZZ_t'/V                 # K = TTT*P_{t|t-1}'ZZ'*(1/V_{t|t-1}) = Kalman gain
            L = TTT - K*ZZ_t

            e = V\dy - K'*r                   # e_t     = (1/V_{t|t-1})dy - K_t'*r_t
            r = ZZ_t'*e + TTT'*r              # r_{t-1} = ZZ'*e_t + TTT'*r_t

            state_disturbances[:,               t] = r
            observable_disturbances[nonmissing, t] = e

        end # of loop backward through this regime's periods

    end # of loop backward through regimes

    # Trim the presample if needed
    if n_presample_periods > 0
        mainsample_periods = n_presample_periods+1:T

        state_disturbances      = state_disturbances[:,      mainsample_periods]
        observable_disturbances = observable_disturbances[:, mainsample_periods]
    end

    return state_disturbances, observable_disturbances
end