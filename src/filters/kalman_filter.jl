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

Returns a `FilterOutput` instance, with the fields:

- `loglh`: a vector of the marginal log likelihoods from t = 1 to Nt
- `s_pred`: `Ns` x `Nt` matrix of one-step predicted state vectors `s_{t|t-1}`
- `P_pred`: `Ns` x `Ns` x `Nt` array of mean squared errors `P_{t|t-1}` of
  predicted state vectors
- `s_filt`: `Ns` x `Nt` matrix of filtered state vectors `s_{t|t}`
- `P_filt`: `Ns` x `Ns` x `Nt` matrix containing mean squared errors `P_{t|t}` of
  filtered state vectors
- `s_0`: `Ns` x 1 initial state vector. This may have reassigned to the last
  presample state vector if `n_presample_periods > 0`
- `P_0`: `Ns` x `Ns` initial state covariance matrix. This may have reassigned to
  the last presample state covariance if `n_presample_periods > 0`
- `s_T`: `Ns` x 1 final filtered state `s_{T|T}`
- `P_T`: `Ns` x `Ns` final filtered state covariance matrix `P_{T|T}`

### Notes

When `s_0` and `P_0` are omitted, they are computed using `init_kalman_filter`.
"""
function kalman_filter(regime_indices::Vector{Range{Int64}}, y::Matrix{S},
    Ts::Vector{Matrix{S}}, Rs::Vector{Matrix{S}}, Cs::Vector{Vector{S}},
    Qs::Vector{Matrix{S}}, Zs::Vector{Matrix{S}}, Ds::Vector{Vector{S}}, Es::Vector{Matrix{S}},
    s_0::Vector{S} = Vector{S}(0), P_0::Matrix{S} = Matrix{S}(0, 0);
    Nt0::Int = 0) where {S<:AbstractFloat}

    # Dimensions
    Nt = size(y, 2) # number of periods of data
    Ns = size(Ts[1], 1) # number of states

    @assert first(regime_indices[1]) == 1
    @assert last(regime_indices[end]) == Nt

    # Initialize array of FilterOutputs
    n_regimes = length(regime_indices)
    fos = Vector{FilterOutput{S}}(n_regimes)

    # Initialize s_0 and P_0
    fi = FilterInput(Ts[1], Rs[1], Cs[1], Qs[1], Zs[1], Ds[1], Es[1])
    s_t, P_t = init_kalman_filter(fi, s_0, P_0)

    # Iterate through regimes
    for i = 1:n_regimes
        y_regime = y[:, regime_indices[i]]
        fos[i] = kalman_filter(y_regime, Ts[i], Rs[i], Cs[i], Qs[i], Zs[i], Ds[i], Es[i],
                               s_t, P_t; Nt0 = 0)
        s_t, P_t = fos[i].s_T, fos[i].P_T
    end

    # Concatenate all filter outputs
    fo = cat(fos...)

    # Remove presample periods from all filter outputs
    remove_presample!(fo, Nt0)
end

function kalman_filter(y::Matrix{S},
    T::Matrix{S}, R::Matrix{S}, C::Vector{S},
    Q::Matrix{S}, Z::Matrix{S}, D::Vector{S}, E::Matrix{S},
    s_0::Vector{S} = Vector{S}(0), P_0::Matrix{S} = Matrix{S}(0, 0);
    Nt0::Int = 0) where {S<:AbstractFloat}

    # Dimensions
    Ns = size(T, 1) # number of states
    Nt = size(y, 2) # number of periods of data

    # Initialize inputs and outputs
    fi = FilterInput(T, R, C, Q, Z, D, E)
    fo = FilterOutput(S, Ns, Nt)

    # Populate s_0 and P_0
    fo.s_0, fo.P_0 = init_kalman_filter(fi, s_0, P_0)

    # Loop through periods t
    for t = 1:Nt
        y_t, fi_t = keep_nonmissing_obs(y[:, t], fi)
        forecast!(fi_t, fo, t)
        update!(fi_t, fo, y_t, t)
    end

    # Populate s_T and P_T
    fo.s_T, fo.P_T = fo.s_filt[:, end], fo.P_filt[:, :, end]

    # Remove presample periods from all filter outputs
    remove_presample!(fo, Nt0)
end

struct FilterInput{S<:AbstractFloat}
    T::Matrix{S}
    R::Matrix{S}
    C::Vector{S}
    Q::Matrix{S}
    Z::Matrix{S}
    D::Vector{S}
    E::Matrix{S}
end

"""
```
keep_nonmissing_obs(y_t, fi::FilterInput)
```

Index out rows of the measurement equation for which we have nonmissing data in
period `t`.
"""
function keep_nonmissing_obs(y_t::Vector{S}, fi::FilterInput{S}) where {S<:AbstractFloat}
    nonnan = .!isnan.(y_t)
    y_t = y_t[nonnan]
    Z_t = fi.Z[nonnan, :]
    D_t = fi.D[nonnan]
    E_t = fi.E[nonnan, :]
    fi_t = FilterInput(fi.T, fi.R, fi.C, fi.Q, Z_t, D_t, E_t)
    return y_t, fi_t
end

mutable struct FilterOutput{S<:AbstractFloat}
    loglh::Vector{S}    # P(y_t | y_{1:t})
    s_pred::Matrix{S}   # s_{t|t-1}
    P_pred::Array{S, 3} # P_{t|t-1}
    s_filt::Matrix{S}   # s_{t|t}
    P_filt::Array{S, 3} # P_{t|t}
    s_0::Vector{S}      # s_{0|0}
    P_0::Matrix{S}      # P_{0|0}
    s_T::Vector{S}      # s_{T|T}
    P_T::Matrix{S}      # P_{T|T}
end

function FilterOutput(S::DataType, Ns::Int, Nt::Int)
    loglh  = zeros(S, Nt)
    s_pred = zeros(S, Ns, Nt)
    P_pred = zeros(S, Ns, Ns, Nt)
    s_filt = zeros(S, Ns, Nt)
    P_filt = zeros(S, Ns, Ns, Nt)
    s_0    = Vector{S}(0)
    P_0    = Matrix{S}(0, 0)
    s_T    = Vector{S}(0)
    P_T    = Matrix{S}(0, 0)
    return FilterOutput(loglh, s_pred, P_pred, s_filt, P_filt, s_0, P_0, s_T, P_T)
end

"""
```
init_kalman_filter(fi::FilterInput, s_0, P_0)
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
function init_kalman_filter(fi::FilterInput{S}, s_0::Vector{S}, P_0::Matrix{S}) where {S<:AbstractFloat}
    T, R, C, Q = fi.T, fi.R, fi.C, fi.Q

    if isempty(s_0) || isempty(P_0)
        e, _ = eig(T)
        if all(abs.(e) .< 1)
            s_0 = (UniformScaling(1) - T)\C
            P_0 = solve_discrete_lyapunov(T, R*Q*R')
        else
            Ns = size(T, 1)
            s_0 = C
            P_0 = 1e6 * eye(Ns)
        end
    end
    return s_0, P_0
end

"""
```
forecast!(fi::FilterInput, fo::FilterOutput, t)
```

Compute the one-step-ahead states s_{t|t-1} and state covariances P_{t|t-1} and
assign to `fo`.
"""
function forecast!(fi::FilterInput{S}, fo::FilterOutput{S}, t::Int) where {S<:AbstractFloat}
    T, R, C, Q = fi.T, fi.R, fi.C, fi.Q
    if t == 1
        s_filt = fo.s_0
        P_filt = fo.P_0
    else
        s_filt = fo.s_filt[:, t-1]
        P_filt = fo.P_filt[:, :, t-1]
    end

    s_pred = T*s_filt + C         # s_{t|t-1} = T*s_{t-1|t-1} + C
    P_pred = T*P_filt*T' + R*Q*R' # P_{t|t-1} = Var s_{t|t-1} = T*P_{t-1|t-1}*T' + R*Q*R'

    fo.s_pred[:, t]    = s_pred
    fo.P_pred[:, :, t] = P_pred
    return s_pred, P_pred
end

"""
```
update!(fi::FilterInput, fo::FilterOutput, y_t, t)
```

Compute the filtered states s_{t|t} and state covariances P_{t|t}, and the
log-likelihood P(y_t | y_{1:t-1}) and assign to `fo`.
"""
function update!(fi::FilterInput{S}, fo::FilterOutput{S}, y_t::Vector{S}, t::Int) where {S<:AbstractFloat}
    Z, D, E = fi.Z, fi.D, fi.E
    s_pred = fo.s_pred[:, t]
    P_pred = fo.P_pred[:, :, t]
    Ny = length(y_t)

    y_pred = Z*s_pred + D         # y_{t|t-1} = Z*s_{t|t-1} + D
    V_pred = Z*P_pred*Z' + E      # V_{t|t-1} = Var y_{t|t-1} = Z*P_{t|t-1}*Z' + E
    V_pred = (V_pred + V_pred')/2
    dy = y_t - y_pred             # dy  = y_t - y_{t|t-1} = prediction error
    ddy = V_pred\dy               # ddy = V_{t|t-1}^{-1}*dy = weighted prediction error

    s_filt = s_pred + P_pred'*Z'*ddy             # s_{t|t} = s_{t|t-1} + P_{t|t-1}'*Z'/V_{t|t-1}*dy
    P_filt = P_pred - P_pred'*Z'/V_pred*Z*P_pred # P_{t|t} = P_{t|t-1} - P_{t|t-1}'*Z'/V_{t|t-1}*Z*P_{t|t-1}
    loglh  = -log(det(V_pred))/2 - first(dy'*ddy/2) - Ny*log(2π)/2

    fo.s_filt[:, t]    = s_filt
    fo.P_filt[:, :, t] = P_filt
    fo.loglh[t]        = loglh
    return s_filt, P_filt, loglh
end

"""
```
remove_presample!(fo::FilterOutput, Nt0)
```

Remove the first `Nt0` periods from all `fo` fields and return.
"""
function remove_presample!(fo::FilterOutput{S}, Nt0::Int) where {S<:AbstractFloat}
    if Nt0 > 0
        Nt = length(fo.loglh)
        insample = (Nt0+1):Nt

        fo.s_0    = fo.s_pred[:,    Nt0]
        fo.P_0    = fo.P_pred[:, :, Nt0]
        fo.loglh  = fo.loglh[insample]
        fo.s_pred = fo.s_pred[:,    insample]
        fo.P_pred = fo.P_pred[:, :, insample]
        fo.s_filt = fo.s_filt[:,    insample]
        fo.P_filt = fo.P_filt[:, :, insample]
    end
    return fo
end

function Base.cat(fo1::FilterOutput{S}, fo2::FilterOutput{S}) where {S<:AbstractFloat}
    loglh  = cat(1, fo1.loglh,  fo2.loglh)
    s_pred = cat(2, fo1.s_pred, fo2.s_pred)
    P_pred = cat(3, fo1.P_pred, fo2.p_pred)
    s_filt = cat(2, fo1.s_filt, fo2.s_filt)
    P_filt = cat(2, fo1.P_filt, fo2.P_filt)
    s_0    = fo1.s_0
    P_0    = fo1.P_0
    s_T    = fo2.s_T
    P_T    = fo2.P_T
    return FilterOutput(loglh, s_pred, P_pred, s_filt, P_filt, s_0, P_0, s_T, P_T)
end

function Base.cat(fos::Vararg{FilterOutput{S}, N}) where {S<:AbstractFloat, N}
    loglh  = cat(1, map(fo -> fo.loglh,  fos)...)
    s_pred = cat(2, map(fo -> fo.s_pred, fos)...)
    P_pred = cat(3, map(fo -> fo.P_pred, fos)...)
    s_filt = cat(2, map(fo -> fo.s_filt, fos)...)
    P_filt = cat(2, map(fo -> fo.P_filt, fos)...)
    s_0    = fos[1].s_0
    P_0    = fos[1].P_0
    s_T    = fos[end].s_T
    P_T    = fos[end].P_T
    return FilterOutput(loglh, s_pred, P_pred, s_filt, P_filt, s_0, P_0, s_T, P_T)
end