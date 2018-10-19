"""
```
koopman_smoother(y, T, R, C, Q, Z, D, E, s_0, P_0, s_pred, P_pred;
    Nt0 = 0)

koopman_smoother(regime_indices, y, Ts, Rs, Cs, Qs, Zs, Ds, Es,
    s_0, P_0, s_pred, P_pred; Nt0 = 0)
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
in the `ϵ_smth` matrix.

The state space is given by:

```
s_{t+1} = C + T*s_t + R*ϵ_t    (transition equation)
y_t     = D + Z*s_t + u_t      (measurement equation)

ϵ_t ∼ N(0, Q)
u_t ∼ N(0, E)
Cov(ϵ_t, u_t) = 0
```

### Inputs

- `y`: `Ny` x `Nt` matrix containing data `y_1, ... , y_T`
- `s_0`: `Ns` x 1 initial state vector
- `P_0`: `Ns` x `Ns` initial state covariance matrix
- `s_pred`: `Ns` x `Nt` matrix of one-step predicted state vectors `s_{t|t-1}`
  (from the Kalman filter)
- `P_pred`: `Ns` x `Ns` x `Nt` array of mean squared errors `P_{t|t-1}` of
  predicted state vectors

**Method 1 only:** state-space system matrices `T`, `R`, `C`, `Q`, `Z`,
`D`, `E`. See `?kalman_filter`

**Method 2 only:** `regime_indices` and system matrices for each regime `Ts`,
`Rs`, `Cs`, `Qs`, `Zs`, `Ds`, `Es`. See `?kalman_filter`

where:

- `Nt`: number of periods for which we have data
- `Ns`: number of states
- `Ne`: number of shocks
- `Ny`: number of observables

### Keyword Arguments

- `Nt0`: if greater than 0, the returned smoothed states and
  shocks matrices will be shorter by that number of columns (taken from the
  beginning)

### Outputs

- `s_smth`: `Ns` x `Nt` matrix of smoothed states `s_{t|T}`
- `ϵ_smth`: `Ne` x `Nt` matrix of smoothed shocks `ϵ_{t|T}`
"""
function koopman_smoother(y::Matrix{Union{S, Missing}},
    T::Matrix{S}, R::Matrix{S}, C::Vector{S}, Q::Matrix{S},
    Z::Matrix{S}, D::Vector{S}, E::Matrix{S},
    s_0::Vector{S}, P_0::Matrix{S}, s_pred::Matrix{S}, P_pred::Array{S, 3};
    Nt0::Int = 0) where {S<:AbstractFloat}

    Nt = size(y, 2)
    koopman_smoother(AbstractRange{Int}[1:Nt], y, Matrix{S}[T], Matrix{S}[R], Vector{S}[C],
        Matrix{S}[Q], Matrix{S}[Z], Vector{S}[D], Matrix{S}[E],
        s_0, P_0, s_pred, P_pred; Nt0 = Nt0)
end

function koopman_smoother(regime_indices::Vector{AbstractRange{Int}}, y::AbstractArray{Union{S, Missing}},
    Ts::Vector{Matrix{S}}, Rs::Vector{Matrix{S}}, Cs::Vector{Vector{S}}, Qs::Vector{Matrix{S}},
    Zs::Vector{Matrix{S}}, Ds::Vector{Vector{S}}, Es::Vector{Matrix{S}},
    s_0::Vector{S}, P_0::Matrix{S}, s_pred::Matrix{S}, P_pred::Array{S, 3};
    Nt0::Int = 0) where {S<:AbstractFloat}

    # Dimensions
    n_regimes = length(regime_indices)
    Nt = size(y, 2)     # number of periods of data
    Ns = size(Ts[1], 1) # number of states
    Ne = size(Rs[1], 2) # number of shocks

    # Call disturbance smoother
    s_dist, _ = koopman_disturbance_smoother(regime_indices, y, Ts, Rs, Qs,
                                             Zs, Ds, Es, s_pred, P_pred; Nt0 = 0)

    # Initialize outputs
    s_smth = zeros(S, Ns, Nt)
    ϵ_smth = zeros(S, Ne, Nt)

    s_t = zeros(S, Ns) # initialize dummy value s.t. s_t is in scope

    for i = 1:n_regimes
        # Get state-space system matrices for this regime
        T, R, Q = Ts[i], Rs[i], Qs[i]

        for t in regime_indices[i]
            r_t = s_dist[:, t]
            s_t = if t == 1
                s_0 + P_0*r_t
            else
                T*s_t + R*Q*R'*r_t
            end

            s_smth[:, t] = s_t
            ϵ_smth[:, t] = Q*R'*r_t
        end
    end

    # Trim the presample if needed
    if Nt0 > 0
        insample = Nt0+1:Nt

        s_smth = s_smth[:, insample]
        ϵ_smth = ϵ_smth[:, insample]
    end

    return s_smth, ϵ_smth
end

"""
```
koopman_disturbance_smoother(y, T, R, Q, Z, D, E, s_pred, P_pred;
    Nt0 = 0)

koopman_smoother(regime_indices, y, Ts, Rs, Qs, Zs, Ds, Es,
    s_pred, P_pred; Nt0 = 0)
```

This disturbance smoother is intended for use with the state smoother
`koopman_smoother` from S.J. Koopman's \"Disturbance Smoother for State Space
Models\" (Biometrika, 1993), as specified in Durbin and Koopman's \"A Simple and
Efficient Simulation Smoother for State Space Time Series Analysis\"
(Biometrika, 2002).

The state space is given by:

```
s_{t+1} = C + T*s_t + R*ϵ_t    (transition equation)
y_t     = D + Z*s_t + u_t      (measurement equation)

ϵ_t ∼ N(0, Q)
u_t ∼ N(0, E)
Cov(ϵ_t, u_t) = 0
```

### Inputs

- `y`: `Ny` x `Nt` matrix containing data `y_1, ... , y_T`
- `s_pred`: `Ns` x `Nt` matrix of one-step predicted state vectors `s_{t|t-1}`
  (from the Kalman filter)
- `P_pred`: `Ns` x `Ns` x `Nt` array of mean squared errors `P_{t|t-1}` of
  predicted state vectors

**Method 1 only:** state-space system matrices `T`, `R`, `Q`, `Z`, `D`, `E`. See
`?kalman_filter`

**Method 2 only:** `regime_indices` and system matrices for each regime `Ts`,
`Rs`, `Qs`, `Zs`, `Ds`, `Es`. See `?kalman_filter`

where:

- `Nt`: number of periods for which we have data
- `Ns`: number of states
- `Ne`: number of shocks
- `Ny`: number of observables

### Keyword Arguments

- `Nt0`: if greater than 0, the returned smoothed disturbances and shocks
  matrices will be shorter by that number of columns (taken from the beginning)

### Outputs

- `s_dist`: `Ns` x `Nt` matrix of transition equation disturbances `r_t`
- `y_dist`: `Ny` x `Nt` matrix of measurement equation disturbances `e_t`
"""
function koopman_disturbance_smoother(y::AbstractArray{Union{S, Missing}},
    T::Matrix{S}, R::Matrix{S}, Q::Matrix{S},
    Z::Matrix{S}, D::Vector{S}, E::Matrix{S},
    s_pred::Matrix{S}, P_pred::Array{S, 3}; Nt0::Int = 0) where {S<:AbstractFloat}

    Nt = size(y, 2)
    koopman_disturbance_smoother(AbstractRange{Int}[1:Nt], y, Matrix{S}[T], Matrix{S}[R],
        Matrix{S}[Q], Matrix{S}[Z], Vector{S}[D], Matrix{S}[E],
        s_pred, P_pred; Nt0 = Nt0)
end

function koopman_disturbance_smoother(regime_indices::Vector{AbstractRange{Int}}, y::AbstractArray{Union{S, Missing}},
    Ts::Vector{Matrix{S}}, Rs::Vector{Matrix{S}}, Qs::Vector{Matrix{S}},
    Zs::Vector{Matrix{S}}, Ds::Vector{Vector{S}}, Es::Vector{Matrix{S}},
    s_pred::Matrix{S}, P_pred::Array{S, 3}; Nt0::Int = 0) where {S<:AbstractFloat}

    # Dimensions
    Nt = size(y ,    2) # number of periods of data
    Ns = size(Ts[1], 1) # number of states
    Ne = size(Rs[1], 2) # number of shocks
    Ny = size(Zs[1], 1) # number of observables

    # Initialize outputs
    s_dist = zeros(S, Ns, Nt)
    y_dist = zeros(S, Ny, Nt)

    r_t = zeros(S, Ns) # r_0 = 0

    for i = length(regime_indices):-1:1
        # Get state-space system matrices for this regime
        T, R, Q = Ts[i], Rs[i], Qs[i]
        Z, D, E = Zs[i], Ds[i], Es[i]

        for t in reverse(regime_indices[i])
            # Keep rows of measurement equation corresponding to non-NaN observables
            nonnan = .!isnan.(y[:, t])
            y_t = y[nonnan, t]
            Z_t = Z[nonnan, :]
            D_t = D[nonnan]
            E_t = E[nonnan, nonnan]

            s_pred_t = s_pred[:, t]            # s_{t|t-1}
            P_pred_t = P_pred[:, :, t]         # P_{t|t-1} = Var s_{t|t-1}

            y_pred_t = Z_t*s_pred_t + D_t      # y_{t|t-1} = Z*s_{t|t-1} + D
            V_pred_t = Z_t*P_pred_t*Z_t' + E_t # V_{t|t-1} = Var y_{t|t-1} = Z*P_{t|t-1}*Z' + E
            dy = y_t - y_pred_t                # dy = y_t - y_{t|t-1} = prediction error
            K = T*P_pred_t*Z_t'/V_pred_t       # K_t = T*P_{t|t-1}'Z'/V_{t|t-1} = Kalman gain

            e_t = V_pred_t\dy - K'*r_t         # e_t     = (1/V_{t|t-1})dy - K_t'*r_t
            r_t = Z_t'*e_t + T'*r_t            # r_{t-1} = Z'*e_t + T'*r_t

            s_dist[:,      t] = r_t
            y_dist[nonnan, t] = e_t
        end # of loop backward through this regime's periods
    end # of loop backward through regimes

    # Trim the presample if needed
    if Nt0 > 0
        insample = Nt0+1:Nt
        s_dist = s_dist[:, insample]
        y_dist = y_dist[:, insample]
    end

    return s_dist, y_dist
end
