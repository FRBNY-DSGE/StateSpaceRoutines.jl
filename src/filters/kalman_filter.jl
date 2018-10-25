using LinearAlgebra
"""
This code is loosely based on a routine originally copyright Federal Reserve Bank of Atlanta
and written by Iskander Karibzhanov.
"""
mutable struct KalmanFilter{S<:AbstractFloat}
    T::Matrix{S}
    R::Matrix{S}
    C::Vector{S}
    Q::Matrix{S}
    Z::Matrix{S}
    D::Vector{S}
    E::Matrix{S}
    s_t::Vector{S} # s_{t|t-1} or s_{t|t}
    P_t::Matrix{S} # P_{t|t-1} or P_{t|t}
    loglh_t::S     # P(y_t | y_{1:t})
end

"""
```
KalmanFilter(T, R, C, Q, Z, D, E, [s_0, P_0])
```

Outer constructor for the `KalmanFilter` type.
"""
function KalmanFilter(T::Matrix{S}, R::Matrix{S}, C::Vector{S}, Q::Matrix{S},
                      Z::Matrix{S}, D::Vector{S}, E::Matrix{S},
                      s_0::Vector{S} = Vector{S}(undef, 0), P_0::Matrix{S} = Matrix{S}(undef, 0, 0)) where {S<:AbstractFloat}
    if isempty(s_0) || isempty(P_0)
        s_0, P_0 = init_stationary_states(T, R, C, Q)
    end

    return KalmanFilter(T, R, C, Q, Z, D, E, s_0, P_0, NaN)
end

"""
```
init_stationary_states(T, R, C, Q)
```

Compute the initial state `s_0` and state covariance matrix `P_0` under the
stationarity condition:

```
s_0  =  (I - T) \\ C
P_0 = reshape(I - kron(T, T)) \\ vec(R*Q*R'), Ns, Ns)
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
function init_stationary_states(T::Matrix{S}, R::Matrix{S}, C::Vector{S},
                                Q::Matrix{S}) where {S<:AbstractFloat}
    e = eigvals(T)
    if all(abs.(e) .< 1)
        s_0 = (UniformScaling(1) - T)\C
        P_0 = solve_discrete_lyapunov(T, R*Q*R')
    else
        Ns = size(T, 1)
        s_0 = C
        P_0 = 1e6 * eye(Ns)
    end
    return s_0, P_0
end

"""
```
kalman_filter(y, T, R, C, Q, Z, D, E, s_0 = Vector(), P_0 = Matrix();
    outputs = [:loglh, :pred, :filt], Nt0 = 0)

kalman_filter(regime_indices, y, Ts, Rs, Cs, Qs, Zs, Ds, Es,
    s_0 = Vector(), P_0 = Matrix(); outputs = [:loglh, :pred, :filt],
    Nt0 = 0)
```

This function implements the Kalman filter for the following state-space model:

```
s_{t+1} = C + T*s_t + R*ϵ_t    (transition equation)
y_t     = D + Z*s_t + u_t      (measurement equation)

ϵ_t ∼ N(0, Q)
u_t ∼ N(0, E)
Cov(ϵ_t, u_t) = 0
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

- `regime_indices`: `Vector{AbstractRange{Int}}` of length `n_regimes`, where
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

- `outputs`: some subset of `[:loglh, :pred, :filt]` specifying which outputs to
  compute and return. There will always be the same number of return values,
  but, for example, `s_pred` and `P_pred` will be returned as empty arrays if
  `:pred` is not in `outputs`
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
`init_stationary_states`.
"""
function kalman_filter(regime_indices::Vector{AbstractRange{Int}}, y::AbstractArray,
    Ts::Vector{Matrix{S}}, Rs::Vector{Matrix{S}}, Cs::Vector{Vector{S}},
    Qs::Vector{Matrix{S}}, Zs::Vector{Matrix{S}}, Ds::Vector{Vector{S}}, Es::Vector{Matrix{S}},
    s_0::Vector{S} = Vector{S}(undef, 0), P_0::Matrix{S} = Matrix{S}(undef, 0, 0);
    outputs::Vector{Symbol} = [:loglh, :pred, :filt],
    Nt0::Int = 0) where {S<:AbstractFloat}

    # Determine outputs
    return_loglh = :loglh in outputs
    return_pred  = :pred  in outputs
    return_filt  = :filt  in outputs

    # Dimensions
    Ns = size(Ts[1], 1) # number of states
    Nt = size(y, 2)     # number of periods of data

    @assert first(regime_indices[1]) == 1
    @assert last(regime_indices[end]) == Nt

    # Initialize inputs and outputs
    k = KalmanFilter(Ts[1], Rs[1], Cs[1], Qs[1], Zs[1], Ds[1], Es[1], s_0, P_0)

    mynan = convert(S, NaN)
    s_pred = return_pred  ? fill(mynan, Ns, Nt)     : Matrix{S}(undef, 0, 0)
    P_pred = return_pred  ? fill(mynan, Ns, Ns, Nt) : Array{S, 3}(undef, 0, 0, 0)
    s_filt = return_filt  ? fill(mynan, Ns, Nt)     : Matrix{S}(undef, 0, 0)
    P_filt = return_filt  ? fill(mynan, Ns, Ns, Nt) : Array{S, 3}(undef, 0, 0, 0)
    loglh  = return_loglh ? fill(mynan, Nt)         : Vector{S}(undef, 0)

    # Populate s_0 and P_0
    s_0 = k.s_t
    P_0 = k.P_t

    # Iterate through regimes
    s_t = s_0
    P_t = P_0

    for i = 1:length(regime_indices)
        ts = regime_indices[i]

        loglh_i, s_pred_i, P_pred_i, s_filt_i, P_filt_i, s_0, P_0, s_t, P_t =
            kalman_filter(y[:, ts], Ts[i], Rs[i], Cs[i], Qs[i], Zs[i], Ds[i], Es[i],
                              s_t, P_t; outputs = outputs, Nt0 = 0)

        if return_loglh
            loglh[ts] = loglh_i
        end
        if return_pred
            s_pred[:,    ts] = s_pred_i
            P_pred[:, :, ts] = P_pred_i
        end
        if return_filt
            s_filt[:,    ts] = s_filt_i
            P_filt[:, :, ts] = P_filt_i
        end
    end

    # Populate s_T and P_T
    s_T, P_T = s_t, P_t

    # Remove presample periods
    loglh, s_pred, P_pred, s_filt, P_filt =
        remove_presample!(Nt0, loglh, s_pred, P_pred, s_filt, P_filt; outputs = outputs)

    return loglh, s_pred, P_pred, s_filt, P_filt, s_0, P_0, s_T, P_T
end

function kalman_filter(y::AbstractArray,
    T::Matrix{S}, R::Matrix{S}, C::Vector{S},
    Q::Matrix{S}, Z::Matrix{S}, D::Vector{S}, E::Matrix{S},
    s_0::Vector{S} = Vector{S}(undef, 0), P_0::Matrix{S} = Matrix{S}(undef, 0, 0);
    outputs::Vector{Symbol} = [:loglh, :pred, :filt],
    Nt0::Int = 0) where {S<:AbstractFloat}

    # Determine outputs
    return_loglh = :loglh in outputs
    return_pred  = :pred  in outputs
    return_filt  = :filt  in outputs

    # Dimensions
    Ns = size(T, 1) # number of states
    Nt = size(y, 2) # number of periods of data

    # Initialize inputs and outputs
    k = KalmanFilter(T, R, C, Q, Z, D, E, s_0, P_0)

    mynan = convert(S, NaN)
    loglh  = return_loglh ? fill(mynan, Nt)         : Vector{S}(undef, 0)
    s_pred = return_pred  ? fill(mynan, Ns, Nt)     : Matrix{S}(undef, 0, 0)
    P_pred = return_pred  ? fill(mynan, Ns, Ns, Nt) : Array{S, 3}(undef, 0, 0, 0)
    s_filt = return_filt  ? fill(mynan, Ns, Nt)     : Matrix{S}(undef, 0, 0)
    P_filt = return_filt  ? fill(mynan, Ns, Ns, Nt) : Array{S, 3}(undef, 0, 0, 0)

    # Populate initial states
    s_0 = k.s_t
    P_0 = k.P_t

    # Loop through periods t
    for t = 1:Nt
        # Forecast
        forecast!(k)
        if return_pred
            s_pred[:,    t] = k.s_t
            P_pred[:, :, t] = k.P_t
        end

        # Update and compute log-likelihood
        update!(k, y[:, t]; return_loglh = return_loglh)
        if return_filt
            s_filt[:,    t] = k.s_t
            P_filt[:, :, t] = k.P_t
        end
        if return_loglh
            loglh[t]        = k.loglh_t
        end

        # Update s_0 and P_0 if Nt0 > 0
        if t == Nt0
            s_0 = k.s_t
            P_0 = k.P_t
        end
    end

    # Populate final states
    s_T = k.s_t
    P_T = k.P_t

    # Remove presample periods
    loglh, s_pred, P_pred, s_filt, P_filt =
        remove_presample!(Nt0, loglh, s_pred, P_pred, s_filt, P_filt; outputs = outputs)

    return loglh, s_pred, P_pred, s_filt, P_filt, s_0, P_0, s_T, P_T
end

"""
```
forecast!(k::KalmanFilter)
```

Compute the one-step-ahead states s_{t|t-1} and state covariances P_{t|t-1} and
assign to `k`.
"""
function forecast!(k::KalmanFilter{S}) where {S<:AbstractFloat}
    T, R, C, Q = k.T, k.R, k.C, k.Q
    s_filt, P_filt = k.s_t, k.P_t

    k.s_t = T*s_filt + C         # s_{t|t-1} = T*s_{t-1|t-1} + C
    k.P_t = T*P_filt*T' + R*Q*R' # P_{t|t-1} = Var s_{t|t-1} = T*P_{t-1|t-1}*T' + R*Q*R'
    return nothing
end

"""
```
update!(k::KalmanFilter{S}, y_obs)
```

Compute the filtered states s_{t|t} and state covariances P_{t|t}, and the
log-likelihood P(y_t | y_{1:t-1}) and assign to `k`.
"""
function update!(k::KalmanFilter{S}, y_obs::AbstractArray;
                 return_loglh::Bool = true) where {S<:AbstractFloat}
    # Keep rows of measurement equation corresponding to non-NaN observables
    nonnan = .!isnan.(y_obs)
    y_obs = y_obs[nonnan]
    Z = k.Z[nonnan, :]
    D = k.D[nonnan]
    E = k.E[nonnan, nonnan]
    Ny = length(y_obs)

    s_pred = k.s_t
    P_pred = k.P_t

    y_pred = Z*s_pred + D         # y_{t|t-1} = Z*s_{t|t-1} + D
    V_pred = Z*P_pred*Z' + E      # V_{t|t-1} = Var y_{t|t-1} = Z*P_{t|t-1}*Z' + E
    V_pred = (V_pred + V_pred')/2
    V_pred_inv = inv(V_pred)
    dy = y_obs - y_pred           # dy = y_t - y_{t|t-1} = prediction error
    PZV = P_pred'*Z'*V_pred_inv

    k.s_t = s_pred + PZV*dy       # s_{t|t} = s_{t|t-1} + P_{t|t-1}'*Z'/V_{t|t-1}*dy
    k.P_t = P_pred - PZV*Z*P_pred # P_{t|t} = P_{t|t-1} - P_{t|t-1}'*Z'/V_{t|t-1}*Z*P_{t|t-1}

    if return_loglh
        k.loglh_t = -(Ny*log(2π) + log(det(V_pred)) + dy'*V_pred_inv*dy)/2 # p(y_t | y_{1:t-1})
    end
    return nothing
end

"""
```
remove_presample!(Nt0, loglh, s_pred, P_pred, s_filt, P_filt)
```

Remove the first `Nt0` periods from all other input arguments and return.
"""
function remove_presample!(Nt0::Int, loglh::Vector{S},
                           s_pred::Matrix{S}, P_pred::Array{S, 3},
                           s_filt::Matrix{S}, P_filt::Array{S, 3};
                           outputs::Vector{Symbol} = [:loglh, :pred, :filt]) where {S<:AbstractFloat}
    if Nt0 > 0
        if :loglh in outputs
            loglh  = loglh[(Nt0+1):end]
        end
        if :pred in outputs
            s_pred = s_pred[:,    (Nt0+1):end]
            P_pred = P_pred[:, :, (Nt0+1):end]
        end
        if :filt in outputs
            s_filt = s_filt[:,    (Nt0+1):end]
            P_filt = P_filt[:, :, (Nt0+1):end]
        end
    end
    return loglh, s_pred, P_pred, s_filt, P_filt
end
