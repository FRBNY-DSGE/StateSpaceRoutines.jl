#=
This code is loosely based on a routine originally copyright Federal Reserve Bank of Atlanta
and written by Iskander Karibzhanov.
=#

"""
```
kalman_filter(y, T, R, C, Q, Z, D, E,
    s_0 = Vector(), P_0 = Matrix(); Nt0 = 0)

kalman_filter(regime_indices, y, Ts, Rs, Cs, Qs, Zs, Ds, Es,
    s_0 = Vector(), P_0 = Matrix(); Nt0 = 0)
```

This function implements the Kalman filter for the following state-space model:

```
s_{t+1} = C + T*s_t + R*ϵ_t    (transition equation)
y_t     = D + Z*s_t + η_t      (measurement equation)

ϵ_t ∼ N(0, Q)
η_t ∼ N(0, E)
Cov(ϵ_t, η_t) = 0
```

### Inputs

- `y`: `Ny` x `Nt` matrix containing data `y_1, ... , y_T`
- `s_0`: optional `Ns` x 1 initial state vector
- `P_0`: optional `Ns` x `Ns` initial state covariance matrix

**Method 1 only:**

- `T`: `Ns` x `Ns` state transition matrix
- `R`: `Ns` x `Ne` matrix in the transition equation mapping shocks to states
- `C`: `Ns` x 1 constant vector in the transition equation
- `Q`: `Ne` x `Ne` matrix of shock covariances
- `Z`: `Ny` x `Ns` matrix in the measurement equation mapping states to
  observables
- `D`: `Ny` x 1 constant vector in the measurement equation
- `E`: `Ny` x `Ny` matrix of measurement error covariances

**Method 2 only:**

- `regime_indices`: `Vector{Range{Int64}}` of length `n_regimes`, where
  `regime_indices[i]` indicates the time periods `t` in regime `i`
- `Ts`: `Vector{Matrix{S}}` of `T` matrices for each regime
- `Rs`
- `Cs`
- `Qs`
- `Zs`
- `Ds`
- `Es`

where:

- `Nt`: number of time periods for which we have data
- `Ns`: number of states
- `Ne`: number of shocks
- `Ny`: number of observables

### Keyword Arguments

- `Nt0`: number of presample periods to omit from all return values

### Outputs

- `loglh`: length `Nt` vector of conditional log-likelihoods P(y_t | y_{1:t-1})
- `s_pred`: `Ns` x `Nt` matrix of one-step predicted state vectors s_{t|t-1}
- `P_pred`: `Ns` x `Ns` x `Nt` array of mean squared errors P_{t|t-1} of
  predicted state vectors
- `s_filt`: `Ns` x `Nt` matrix of filtered state vectors s_{t|t}
- `P_filt`: `Ns` x `Ns` x `Nt` matrix containing mean squared errors P_{t|t} of
  filtered state vectors
- `s_0`: `Ns` x 1 initial state vector. This may have been reassigned to the
  last presample state vector if `Nt0 > 0`
- `P_0`: `Ns` x `Ns` initial state covariance matrix. This may have been
  reassigned to the last presample state covariance if `Nt0 > 0`
- `s_T`: `Ns` x 1 final filtered state `s_{T|T}`
- `P_T`: `Ns` x `Ns` final filtered state covariance matrix `P_{T|T}`

### Notes

When `s_0` and `P_0` are omitted, they are computed using
`init_kalman_filter_states`.
"""
function kalman_filter(regime_indices::Vector{Range{Int64}}, y::Matrix{S},
    Ts::Vector{Matrix{S}}, Rs::Vector{Matrix{S}}, Cs::Vector{Vector{S}},
    Qs::Vector{Matrix{S}}, Zs::Vector{Matrix{S}}, Ds::Vector{Vector{S}}, Es::Vector{Matrix{S}},
    s_0::Vector{S} = Vector{S}(0), P_0::Matrix{S} = Matrix{S}(0, 0);
    outputs::Vector{Symbol} = [:loglh, :pred, :filt],
    Nt0::Int = 0) where {S<:AbstractFloat}

    # Dimensions
    Ns = size(Ts[1], 1) # number of states
    Nt = size(y, 2)     # number of periods of data

    @assert first(regime_indices[1]) == 1
    @assert last(regime_indices[end]) == Nt

    # Initialize outputs
    loglh, s_pred, P_pred, s_filt, P_filt = init_kalman_filter_outputs(S, Ns, Nt; outputs = outputs)

    # Populate s_0 and P_0
    s_0, P_0 = init_kalman_filter_states(s_0, P_0, Ts[1], Rs[1], Cs[1], Qs[1])
    s_t, P_t = s_0, P_0

    # Iterate through regimes
    for i = 1:length(regime_indices)
        ts = regime_indices[i]
        loglh_i, s_pred_i, P_pred_i, s_filt_i, P_filt_i, _, _, s_t, P_t =
            kalman_filter(y[:, ts], Ts[i], Rs[i], Cs[i], Qs[i], Zs[i], Ds[i], Es[i],
                              s_t, P_t; outputs = outputs, Nt0 = 0)
        if :loglh in outputs
            loglh[ts] = loglh_i
        end
        if :pred in outputs
            s_pred[:,    ts] = s_pred_i
            P_pred[:, :, ts] = P_pred_i
        end
        if :filt in outputs
            s_filt[:,    ts] = s_filt_i
            P_filt[:, :, ts] = P_filt_i
        end
    end

    # Populate s_T and P_T
    s_T, P_T = s_t, P_t

    # Remove presample periods from all filter outputs
    loglh, s_pred, P_pred, s_filt, P_filt, s_0, P_0, s_T, P_T =
        remove_presample!(Nt0, loglh, s_pred, P_pred, s_filt, P_filt, s_0, P_0, s_T, P_T;
                          outputs = outputs)
end

function kalman_filter(y::Matrix{S},
    T::Matrix{S}, R::Matrix{S}, C::Vector{S},
    Q::Matrix{S}, Z::Matrix{S}, D::Vector{S}, E::Matrix{S},
    s_0::Vector{S} = Vector{S}(0), P_0::Matrix{S} = Matrix{S}(0, 0);
    outputs::Vector{Symbol} = [:loglh, :pred, :filt],
    Nt0::Int = 0) where {S<:AbstractFloat}

    # Dimensions
    Ns = size(T, 1) # number of states
    Nt = size(y, 2) # number of periods of data

    # Initialize outputs
    loglh, s_pred, P_pred, s_filt, P_filt = init_kalman_filter_outputs(S, Ns, Nt; outputs = outputs)

    # Populate s_0 and P_0
    s_0, P_0 = init_kalman_filter_states(s_0, P_0, T, R, C, Q)
    s_t, P_t = s_0, P_0

    for t = 1:Nt
        # Forecast
        s_t, P_t = forecast!(s_t, P_t, T, R, C, Q)
        if :pred in outputs
            s_pred[:, t], P_pred[:, :, t] = s_t, P_t
        end

        # Update
        s_t, P_t, loglh_t = update!(s_t, P_t, y[:, t], Z, D, E; compute_loglh = :loglh in outputs)
        if :filt in outputs
            s_filt[:, t], P_filt[:, :, t] = s_t, P_t
        end
        if :loglh in outputs
            loglh[t] = loglh_t
        end
    end

    # Populate s_T and P_T
    s_T, P_T = s_t, P_t

    # Remove presample periods from all filter outputs
    loglh, s_pred, P_pred, s_filt, P_filt, s_0, P_0, s_T, P_T =
        remove_presample!(Nt0, loglh, s_pred, P_pred, s_filt, P_filt, s_0, P_0, s_T, P_T;
                          outputs = outputs)
end

function init_kalman_filter_outputs(S::DataType, Ns::Int, Nt::Int;
                                    outputs::Vector{Symbol} = [:loglh, :pred, :filt])
    if :loglh in outputs
        loglh = zeros(S, Nt)
    else
        loglh = Vector{S}(0)
    end
    if :pred in outputs
        s_pred = zeros(S, Ns, Nt)
        P_pred = zeros(S, Ns, Ns, Nt)
    else
        s_pred = Matrix{S}(0, 0)
        P_pred = Array{S, 3}(0, 0, 0)
    end
    if :filt in outputs
        s_filt = zeros(S, Ns, Nt)
        P_filt = zeros(S, Ns, Ns, Nt)
    else
        s_filt = Matrix{S}(0, 0)
        P_filt = Array{S, 3}(0, 0, 0)
    end
    return loglh, s_pred, P_pred, s_filt, P_filt
end

"""
```
init_kalman_filter_states(s_0, P_0, T, R, C, Q)
```

Compute the initial state vector and its covariance matrix of the time invariant
Kalman filters under the stationarity condition:

```
s_0  = (I - T)\C
P_0 = reshape(I - kron(T, T))\vec(R*Q*R'), Ns, Ns)
```

where:

- `kron(T, T)` is a matrix of dimension `Ns^2` x `Ns^2`, the Kronecker
  product of `T`
- `vec(R*Q*R')` is the `Ns^2` x 1 column vector constructed by stacking the
  `Ns` columns of `R*Q*R'`

All eigenvalues of `T` are inside the unit circle when the state space model
is stationary. When the preceding formula cannot be applied, the initial state
vector estimate is set to `C` and its covariance matrix is given by `1e6 * I`.
"""
function init_kalman_filter_states(s_0::Vector{S}, P_0::Matrix{S},
                                   T::Matrix{S}, R::Matrix{S}, C::Vector{S},
                                   Q::Matrix{S}) where {S<:AbstractFloat}
    if isempty(s_0) || isempty(P_0)
        F::Base.LinAlg.Eigen{S, S, Matrix{S}, Vector{S}} = eigfact(T)
        e = F.values
        if all(abs.(e) .< 1)
            s_0 = (UniformScaling(1) - T)\C
            P_0::Matrix{S} = solve_discrete_lyapunov(T, R*Q*R')
        else
            Ns = size(T, 1)
            s_0 = C
            P_0 = 1e6 * eye(S, Ns)
        end
    end
    return s_0, P_0
end

"""
```
forecast!(s_filt, P_filt, T, R, C, Q)
```

Compute and return the one-step-ahead predicted states s_{t|t-1} and state
covariances P_{t|t-1}.
"""
function forecast!(s_filt::Vector{S}, P_filt::Matrix{S},
                   T::Matrix{S}, R::Matrix{S}, C::Vector{S},
                   Q::Matrix{S}) where {S<:AbstractFloat}
    s_pred = T*s_filt + C         # s_{t|t-1} = T*s_{t-1|t-1} + C
    P_pred = T*P_filt*T' + R*Q*R' # P_{t|t-1} = Var s_{t|t-1} = T*P_{t-1|t-1}*T' + R*Q*R'
    return s_pred, P_pred
end

"""
```
update!(s_pred, P_pred, y_obs, Z, D, E)
```

Compute and return the filtered states s_{t|t} and state covariances P_{t|t},
and the log-likelihood P(y_t | y_{1:t-1}).
"""
function update!(s_pred::Vector{S}, P_pred::Matrix{S}, y_obs::Vector{S},
                 Z::Matrix{S}, D::Vector{S}, E::Matrix{S};
                 compute_loglh::Bool = true) where {S<:AbstractFloat}
    # Index out rows of the measurement equation for which we have nonmissing
    # data in period t
    nonnan = .!isnan.(y_obs)
    y_obs = y_obs[nonnan]
    Z = Z[nonnan, :]
    D = D[nonnan]
    E = E[nonnan, nonnan]
    Ny = length(y_obs)

    y_pred = Z*s_pred + D         # y_{t|t-1} = Z*s_{t|t-1} + D
    V_pred = Z*P_pred*Z' + E      # V_{t|t-1} = Var y_{t|t-1} = Z*P_{t|t-1}*Z' + E
    V_pred = (V_pred + V_pred')/2
    dy = y_obs - y_pred           # dy  = y_t - y_{t|t-1} = prediction error
    ddy = V_pred\dy               # ddy = V_{t|t-1}^{-1}*dy = weighted prediction error

    s_filt = s_pred + P_pred'*Z'*ddy             # s_{t|t} = s_{t|t-1} + P_{t|t-1}'*Z'/V_{t|t-1}*dy
    P_filt = P_pred - P_pred'*Z'/V_pred*Z*P_pred # P_{t|t} = P_{t|t-1} - P_{t|t-1}'*Z'/V_{t|t-1}*Z*P_{t|t-1}
    loglh  = if compute_loglh
        -(Ny*log(2π) + log(det(V_pred)) + dy'*ddy)/2
    else
        NaN
    end
    return s_filt, P_filt, loglh
end

"""
```
remove_presample!(Nt0, loglh, s_pred, P_pred, s_filt, P_filt, s_0, P_0,
    s_T, P_T)
```

Remove the first `Nt0` periods from all other input arguments and return.
"""
function remove_presample!(Nt0::Int, loglh::Vector{S},
                           s_pred::Matrix{S}, P_pred::Array{S, 3},
                           s_filt::Matrix{S}, P_filt::Array{S, 3},
                           s_0::Vector{S}, P_0::Matrix{S},
                           s_T::Vector{S}, P_T::Matrix{S};
                           outputs::Vector{Symbol} = [:loglh, :pred, :filt]) where {S<:AbstractFloat}
    if Nt0 > 0
        Nt = length(loglh)
        insample = (Nt0+1):Nt

        if :loglh in outputs
            loglh  = loglh[insample]
        end
        if :pred in outputs
            s_pred = s_pred[:,    insample]
            P_pred = P_pred[:, :, insample]
        end
        if :filt in outputs
            s_filt = s_filt[:,    insample]
            P_filt = P_filt[:, :, insample]
        end
        s_0    = s_pred[:,    Nt0]
        P_0    = P_pred[:, :, Nt0]
    end
    return loglh, s_pred, P_pred, s_filt, P_filt, s_0, P_0, s_T, P_T
end
