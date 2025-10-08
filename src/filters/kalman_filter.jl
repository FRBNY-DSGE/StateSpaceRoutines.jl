using LinearAlgebra
"""
This code is loosely based on a routine originally copyright Federal Reserve Bank of Atlanta
and written by Iskander Karibzhanov.
"""
mutable struct KalmanFilter{S<:Number}
    T::AbstractMatrix{S}
    R::AbstractMatrix{S}
    C::AbstractVector{S}
    Q::AbstractMatrix{S}
    Z::AbstractMatrix{S}
    D::AbstractVector{S}
    E::AbstractMatrix{S}
    s_t::AbstractVector{S} # s_{t|t-1} or s_{t|t}
    P_t::AbstractMatrix{S} # P_{t|t-1} or P_{t|t}
    loglh_t::U where {U<:Number}     # P(y_t | y_{1:t})
    converged::Bool
    PZV::AbstractMatrix{S}
end

"""
```
KalmanFilter(T, R, C, Q, Z, D, E, [s_0, P_0])
```

Outer constructor for the `KalmanFilter` type.
"""
function KalmanFilter(T::AbstractMatrix{S}, R::AbstractMatrix{S}, C::AbstractVector{S}, Q::AbstractMatrix{S},
                      Z::AbstractMatrix{S}, D::AbstractVector{S}, E::AbstractMatrix{S},
                      s_0::AbstractVector{S} = Vector{S}(undef, 0),
                      P_0::AbstractMatrix{S} = Matrix{S}(undef, 0, 0),
                      converged::Bool = false,
                      PZV::AbstractMatrix{S} = Matrix{S}(undef, 0, 0)) where {S<:Real}
    if isempty(s_0) || isempty(P_0)
        if issparse(T) # can't call eigvals on a sparse matrix
            s_0, P_0 = init_stationary_states(Matrix(T), R, C, Q)
        else
            s_0, P_0 = init_stationary_states(T, R, C, Q)
        end
    end

    return KalmanFilter(T, R, C, Q, Z, D, E, s_0, P_0, NaN, converged, PZV)
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
function init_stationary_states(T::AbstractMatrix{S}, R::AbstractMatrix{S}, C::AbstractVector{S},
                                Q::AbstractMatrix{S};
                                check_is_stationary::Bool = true) where {S<:Real}
    if check_is_stationary
        e = eigvals(T)
        if all(abs.(e) .< 1)
            s_0 = (UniformScaling(1) - T)\C
            P_0 = solve_discrete_lyapunov(T, R*Q*R')
        else
            Ns = size(T, 1)
            s_0 = C
            P_0 = 1e6 * Matrix(1.0I, Ns, Ns)
        end
    else
        s_0 = (Matrix{eltype(T)}(I, size(T)...) - T)\C
        P_0 = solve_discrete_lyapunov(T, R*Q*R')
    end
    return s_0, P_0
end

function init_stationary_states(T::TrackedMatrix{S}, R::TrackedMatrix{S}, C::TrackedVector{S},
                                Q::TrackedMatrix{S};
                                check_is_stationary::Bool = true) where {S<:Real}
    mat_type = return_tracker_parameter_type(T)
    if check_is_stationary
        e = eigvals(T)
        if all(abs.(e) .< 1)
            s_0 = (UniformScaling(1) - T)\C
            P_0 = solve_discrete_lyapunov(T, R*Q*R')
        else
            Ns = size(T, 1)
            s_0 = C
            P_0 = 1e6 * Matrix(1.0I, Ns, Ns)
        end
    else
        s_0 = (Matrix{mat_type}(I, size(T)...) - T)\C
        P_0 = solve_discrete_lyapunov(T, R*Q*R')
    end
    return s_0, P_0
end

"""
```
kalman_filter(y, T, R, C, Q, Z, D, E, s_0 = AbstractVector(), P_0 = Matrix();
    outputs = [:loglh, :pred, :filt], Nt0 = 0)

kalman_filter(regime_indices, y, Ts, Rs, Cs, Qs, Zs, Ds, Es,
    s_0 = AbstractVector(), P_0 = Matrix(); outputs = [:loglh, :pred, :filt],
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

- `regime_indices`: `AbstractVector{UnitRange{Int}}` of length `n_regimes`, where
  `regime_indices[i]` indicates the time periods `t` in regime `i`
- `Ts`: `AbstractVector{Matrix{S}}` of `T` matrices for each regime
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
function kalman_filter(regime_indices::Vector{UnitRange{Int}}, y::AbstractArray,
                       Ts::Vector{<:AbstractMatrix{S}}, Rs::Vector{<:AbstractMatrix{S}},
                       Cs::Vector{<:AbstractVector{S}}, Qs::Vector{<:AbstractMatrix{S}},
                       Zs::Vector{<:AbstractMatrix{S}}, Ds::Vector{<:AbstractVector{S}},
                       Es::Vector{<:AbstractMatrix{S}}, s_0::AbstractVector{S} = Vector{S}(undef, 0),
                       P_0::AbstractMatrix{S} = Matrix{S}(undef, 0, 0);
                       outputs::AbstractVector{Symbol} = [:loglh, :pred, :filt],
                       Nt0::Int = 0, tol::AbstractFloat = 0.0) where {S<:Real}

    # Determine outputs
    return_loglh = :loglh in outputs
    return_pred  = :pred  in outputs
    return_filt  = :filt  in outputs

    # Dimensions
    Ns = size(Ts[1], 1) # number of states
    Nt = size(y, 2)     # number of periods of data

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
                          s_t, P_t; outputs = outputs, Nt0 = 0, tol = tol)

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

function kalman_filter(y::AbstractArray, T::AbstractMatrix{S}, R::AbstractMatrix{S},
                       C::AbstractVector{S}, Q::AbstractMatrix{S}, Z::AbstractMatrix{S},
                       D::AbstractVector{S}, E::AbstractMatrix{S},
                       s_0::AbstractVector{S} = Vector{S}(undef, 0),
                       P_0::AbstractMatrix{S} = Matrix{S}(undef, 0, 0);
                       outputs::AbstractVector{Symbol} = [:loglh, :pred, :filt],
                       Nt0::Int = 0, tol::AbstractFloat = 0.0) where {S<:Real}

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
    RQR′ = R * Q * R'
    for t = 1:Nt
        # Forecast
        forecast!(k, RQR′) # calculate RQR′ once for efficiency
        if return_pred
            s_pred[:,    t] = k.s_t
            P_pred[:, :, t] = k.P_t
        end

        # Update and compute log-likelihood
        update!(k, y[:, t]; return_loglh = return_loglh, tol = tol)
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



function kalman_likelihood(regime_indices::Vector{UnitRange{Int}}, y::AbstractArray,
                           Ts::Vector{<:AbstractMatrix{S}}, Rs::Vector{<:AbstractMatrix{S}},
                           Cs::Vector{<:AbstractVector{S}}, Qs::Vector{<:AbstractMatrix{S}},
                           Zs::Vector{<:AbstractMatrix{S}}, Ds::Vector{<:AbstractVector{S}},
                           Es::Vector{<:AbstractMatrix{S}}, s_0::AbstractVector{S} = Vector{S}(undef, 0),
                           P_0::AbstractMatrix{S} = Matrix{S}(undef, 0, 0);
                           add_zlb_duration::Tuple{Bool, Int} = (false, 1),
                           Nt0::Int = 0, tol::AbstractFloat = 0.0, switching::Bool = true) where {S<:Real}
    # Dimensions
    Nt = size(y, 2) # number of periods of data

    @assert first(regime_indices[1]) == 1
    @assert last(regime_indices[end]) == Nt

    # Initialize inputs and outputs
    k = KalmanFilter(Ts[1], Rs[1], Cs[1], Qs[1], Zs[1], Ds[1], Es[1], s_0, P_0)

    mynan = convert(S, NaN)
    loglh = fill(mynan, Nt)

    # Iterate through regimes
    s_t = k.s_t
    P_t = k.P_t

    if add_zlb_duration[1]
        zlb_st = similar(s_t)
    end

    if length(regime_indices) == 1
        if add_zlb_duration[1]
            loglh[1:add_zlb_duration[2]], s_t, P_t = kalman_likelihood(y[:, 1:add_zlb_duration[2]], Ts[1], Rs[1], Cs[1], Qs[1],
                                      Zs[1], Ds[1], Es[1], s_t, P_t;
                                      Nt0 = 0, tol = tol, switching = true)

            zlb_st .= s_t

            loglh[add_zlb_duration[2]+1:end] = kalman_likelihood(y[:, add_zlb_duration[2]+1:end], Ts[1], Rs[1], Cs[1], Qs[1],
                                      Zs[1], Ds[1], Es[1], s_t, P_t;
                                      Nt0 = 0, tol = tol, switching = false)
        else
            
            loglh = kalman_likelihood(y ,Ts[1],Rs[1], Cs[1], Qs[1],
                                      Zs[1], Ds[1], Es[1], s_t, P_t;
                                      Nt0 = 0, tol = tol, switching = false)
        end
    else
        if add_zlb_duration[1]
            zlb_ind = findfirst(x -> add_zlb_duration[2] in x, regime_indices)
            if add_zlb_duration[2] != regime_indices[zlb_ind][end]
                insert!(regime_indices, (add_zlb_duration[2]+1):regime_indices[zlb_ind][end], zlb_ind+1)
                regime_indices[zlb_ind] = regime_indices[zlb_ind][1]:add_zlb_duration[2]
            end

            for i = 1:length(regime_indices)
                ts = regime_indices[i]
                loglh[ts], s_t, P_t = kalman_likelihood(y[:, ts], Ts[i], Rs[i], Cs[i], Qs[i],
                                                        Zs[i], Ds[i], Es[i], s_t, P_t;
                                                        Nt0 = 0, tol = tol, switching = true)

                if length(ts) > 0 && add_zlb_duration[2] == ts[end]
                    zlb_st .= s_t
                end
            end
        else
            for i = 1:length(regime_indices)
                ts = regime_indices[i]
                loglh[ts], s_t, P_t = kalman_likelihood(y[:, ts], Ts[i], Rs[i], Cs[i], Qs[i],
                                                        Zs[i], Ds[i], Es[i], s_t, P_t;
                                                        Nt0 = 0, tol = tol, switching = true)
            end
        end
    end

    # Remove presample periods
    loglh = remove_presample!(Nt0, loglh)

    if add_zlb_duration[1]
        return loglh, zlb_st
    else
        return loglh
    end
end

#=
function kalman_likelihood(regime_indices::Vector{UnitRange{Int}}, y::AbstractArray,
                           Ts::Vector{<:AbstractMatrix{S}}, Rs::Vector{<:AbstractMatrix{S}},
                           Cs::Vector{<:AbstractVector{S}}, Qs::Vector{<:AbstractMatrix{S}},
                           Zs::Vector{<:AbstractMatrix{S}}, Ds::Vector{<:AbstractVector{S}},
                           Es::Vector{<:AbstractMatrix{S}}, s_0::AbstractVector{S} = Vector{S}(undef, 0),
                           P_0::AbstractMatrix{S} = Matrix{S}(undef, 0, 0);
                           add_zlb_duration::Tuple{Bool, Int} = (false, 1),
                           Nt0::Int = 0, tol::AbstractFloat = 0.0) where {S<:Real}
    # Dimensions
    Nt = size(y, 2) # number of periods of data

    @assert first(regime_indices[1]) == 1
    @assert last(regime_indices[end]) == Nt

    # Initialize inputs and outputs
    k = KalmanFilter(Ts[1], Rs[1], Cs[1], Qs[1], Zs[1], Ds[1], Es[1], s_0, P_0)

    mynan = convert(S, NaN)
    loglh = fill(mynan, Nt)

    # Iterate through regimes
    s_t = k.s_t
    P_t = k.P_t

    if length(regime_indices) == 1
        if add_zlb_duration[1]
            loglh[1:add_zlb_duration[2]], s_t, P_t = kalman_likelihood(y[:, 1:add_zlb_duration[2]], Ts[1], Rs[1], Cs[1], Qs[1],
                                      Zs[1], Ds[1], Es[1], s_t, P_t;
                                      Nt0 = 0, tol = tol, switching = true)

            ## Compute implied ZLB duration
            ### Save settings that need to change to forecast from add_zlb_duration[2]
            horizons = get_setting(m, :forecast_horizons)
            orig_regime_eqcond_info = get_setting(m, :regime_eqcond_info)
            orig_reg_forecast_start = get_setting(m, :reg_forecast_start)
            orig_reg_post_conditional_end = get_setting(m, :reg_post_conditional_end)
            orig_n_fcast_regimes = get_setting(m, :n_fcast_regimes)
            orig_n_hist_regimes = get_setting(m, :n_hist_regimes)
            orig_min_temp_altpol_len = haskey(m.settings, :min_temporary_altpolicy_length) ? get_setting(m, :min_temporary_altpolicy_length) : nothing
            orig_max_temp_altpol_len = haskey(m.settings, :max_temporary_altpolicy_length) ? get_setting(m, :max_temporary_altpolicy_length) : nothing
            orig_hist_temp_altpol_len = haskey(m.settings, :historical_temporary_altpolicy_length) ? get_setting(m, :historical_temporary_altpolicy_length) : nothing
            orig_cred_vary_until = haskey(m.settings, :cred_vary_until) ? get_setting(m, :cred_vary_until) : nothing

            ### Reset settings for add_zlb_duration[2]
            for i in
            get_setting(m, :regime_eqcond_info)[i].weights =
weights now fixed to 2020Q4, remove ZLB in main policy?
            reg_forecast_start = use regime_inds and the ts that's passed in
            reg_post_conditional_end = reg_forecast_start
            n_fcast_regimes = n_regimes - reg_forecast_start + 1
            n_hist_regimes = n_regimes - n_fcast_regimes
            min_temp_altpol_len (if exists) = 0
            max_temp_altpol_len (if exists) = remove
            historical_temporary_altpolicy_length = n_hist_regimes - 1 (this is a little bit of hard coding)
            cred_vary_until = n_regimes


            forecast(m, s_t, zeros(length(s_t), horizons), zeros(length(m.observables), horizons),
                     zeros(length(m.pseudo_observables), horizons), zeros(length(m.exogenous_shocks), horizons);
                     cond_type = :none)

            ## Compute loss for ZLB duration

            loglh[add_zlb_duration[2]+1:end] = kalman_likelihood(y[:, add_zlb_duration[2]+1:end], Ts[1], Rs[1], Cs[1], Qs[1],
                                      Zs[1], Ds[1], Es[1], s_t, P_t;
                                      Nt0 = 0, tol = tol, switching = false)
        else
            loglh = kalman_likelihood(y, Ts[1], Rs[1], Cs[1], Qs[1],
                                      Zs[1], Ds[1], Es[1], s_t, P_t;
                                      Nt0 = 0, tol = tol, switching = false)
        end
    elseif !add_zlb_duration[1]
        for i = 1:length(regime_indices)
            ts = regime_indices[i]
            loglh[ts], s_t, P_t = kalman_likelihood(y[:, ts], Ts[i], Rs[i], Cs[i], Qs[i],
                                                    Zs[i], Ds[i], Es[i], s_t, P_t;
                                                    Nt0 = 0, tol = tol, switching = true)
        end
    else
        zlb_ind = findfirst(x -> add_zlb_duration[2] in x, regime_indices)
        if add_zlb_duration[2] != regime_indices[zlb_ind][end]
            insert!(regime_indices, (add_zlb_duration[2]+1):regime_indices[zlb_ind][end], zlb_ind+1)
            regime_indices[zlb_ind] = regime_indices[zlb_ind][1]:add_zlb_duration[2]
        end

        for i = 1:length(regime_indices)
            ts = regime_indices[i]
            loglh[ts], s_t, P_t = kalman_likelihood(y[:, ts], Ts[i], Rs[i], Cs[i], Qs[i],
                                                    Zs[i], Ds[i], Es[i], s_t, P_t;
                                                    Nt0 = 0, tol = tol, switching = true)

            if add_zlb_duration[2] == ts[end]
                ## Compute implied ZLB duration
                ### Save settings that need to change to forecast from add_zlb_duration[2]
                horizons = get_setting(m, :forecast_horizons)
                orig_regime_eqcond_info = get_setting(m, :regime_eqcond_info)
                orig_reg_forecast_start = get_setting(m, :reg_forecast_start)
                orig_reg_post_conditional_end = get_setting(m, :reg_post_conditional_end)
                orig_n_fcast_regimes = get_setting(m, :n_fcast_regimes)
                orig_n_hist_regimes = get_setting(m, :n_hist_regimes)
                orig_min_temp_altpol_len = haskey(m.settings, :min_temporary_altpolicy_length) ? get_setting(m, :min_temporary_altpolicy_length) : nothing
                orig_max_temp_altpol_len = haskey(m.settings, :max_temporary_altpolicy_length) ? get_setting(m, :max_temporary_altpolicy_length) : nothing
                orig_hist_temp_altpol_len = haskey(m.settings, :historical_temporary_altpolicy_length) ? get_setting(m, :historical_temporary_altpolicy_length) : nothing
                orig_cred_vary_until = haskey(m.settings, :cred_vary_until) ? get_setting(m, :cred_vary_until) : nothing

                ### Reset settings for add_zlb_duration[2]
                for a in i+1:length(get_setting(m, :regime_eqcond_info))
                    get_setting(m, :regime_eqcond_info)[a].weights = get_setting(m, :regime_eqcond_info)[i].weights
                    get_setting(m, :regime_eqcond_info)[a].alternative_policy = DSGE.flexible_ait()
weights now fixed to 2020Q4, remove ZLB in main policy?
            reg_forecast_start = use regime_inds and the ts that's passed in
            reg_post_conditional_end = reg_forecast_start
            n_fcast_regimes = n_regimes - reg_forecast_start + 1
            n_hist_regimes = n_regimes - n_fcast_regimes
            min_temp_altpol_len (if exists) = 0
            max_temp_altpol_len (if exists) = remove
            historical_temporary_altpolicy_length = n_hist_regimes - 1 (this is a little bit of hard coding)
            cred_vary_until = n_regimes


            forecast(m, s_t, zeros(length(s_t), horizons), zeros(length(m.observables), horizons),
                     zeros(length(m.pseudo_observables), horizons), zeros(length(m.exogenous_shocks), horizons);
                     cond_type = :none)

            ## Compute loss for ZLB duration

            end
        end
    end

    # Remove presample periods
    loglh = remove_presample!(Nt0, loglh)

    return loglh
end
=#

function kalman_likelihood(y::AbstractArray, T::AbstractMatrix{S}, R::AbstractMatrix{S}, C::AbstractVector{S},
                            Q::AbstractMatrix{S}, Z::AbstractMatrix{S}, D::AbstractVector{S}, E::AbstractMatrix{S},
                            s_0::AbstractVector{S} = Vector{S}(undef, 0),
                            P_0::AbstractMatrix{S} = Matrix{S}(undef, 0, 0);
                            Nt0::Int = 0, tol::AbstractFloat = 0.0,
                            switching::Bool = true) where {S<:Real}

    kalman_likelihood(y , T, R, C, Q, Z, D, E;
                            s_0 = s_0,
                            P_0 = P_0,
                            Nt0 = Nt0, tol = tol,
                            switching = switching)

end

function kalman_likelihood(y::AbstractArray, T::AbstractMatrix{S}, R::AbstractMatrix{S}, C::AbstractVector{S},
                           Q::AbstractMatrix{S}, Z::AbstractMatrix{S}, D::AbstractVector{S}, E::AbstractMatrix{S};
                           s_0::AbstractVector{S} = Vector{S}(undef, 0),
                           P_0::AbstractMatrix{S} = Matrix{S}(undef, 0, 0),
                           Nt0::Int = 0, tol::AbstractFloat = 0.0,
                           switching::Bool = true) where {S<:Real}
    
    # Dimensions
    Nt = size(y, 2) # number of periods of data

    # Initialize inputs and outputs
    k = KalmanFilter(T, R, C, Q, Z, D, E, s_0, P_0)

    mynan = convert(S, NaN)
    loglh = fill(mynan, Nt)

    # Loop through periods t
    RQR′ = R * Q * R'
    for t = 1:Nt
        # Forecast
        forecast!(k, RQR′) # calculate RQR′ once for efficiency

        # Update and compute log-likelihood
        update!(k, y[:, t]; return_loglh = true, tol = tol)
        loglh[t] = k.loglh_t
    end

    # Remove presample periods
    loglh = remove_presample!(Nt0, loglh)

    if switching
        return loglh, k.s_t, k.P_t
    else
        return loglh
    end
end

function kalman_likelihood(y::AbstractArray, T::TrackedMatrix{S}, R::TrackedMatrix{S},
                           C::TrackedVector{S}, Q::TrackedMatrix{S}, Z::TrackedMatrix{S},
                           D::TrackedVector{S}, E::TrackedMatrix{S},
                           s_0::TrackedVector{S} = param(Vector{S}(undef, 0)),
                           P_0::TrackedMatrix{S} = param(Matrix{S}(undef, 0, 0));
                           Nt0::Int = 0, tol::AbstractFloat = 0.0) where {S<:Real}
    # Dimensions
    Nt = size(y, 2) # number of periods of data

    # Initialize inputs and outputs
    k = KalmanFilter(T, R, C, Q, Z, D, E, s_0, P_0)

    mynan = convert(S, NaN)
    loglh = fill(Tracker.TrackedReal{S}(mynan), Nt)

    # Loop through periods t
    RQR′ = R * Q * R'
    for t = 1:Nt
        # Forecast
        forecast!(k, RQR′) # calculate RQR′ once for efficiency

        # Update and compute log-likelihood
        update!(k, y[:, t]; return_loglh = true, tol = tol)
        loglh[t] = k.loglh_t
    end

    # Remove presample periods
    loglh = remove_presample!(Nt0, loglh)

    return loglh
end


"""
```
forecast!(k::KalmanFilter)
```

Compute the one-step-ahead states s_{t|t-1} and state covariances P_{t|t-1} and
assign to `k`.
"""
function forecast!(k::KalmanFilter{S}, RQR′::AbstractMatrix = k.R * k.Q * k.R') where {S<:Real}
    T, C           = k.T, k.C
    s_filt, P_filt = k.s_t, k.P_t

    k.s_t = T*s_filt + C         # s_{t|t-1} = T*s_{t-1|t-1} + C
    k.P_t = T*P_filt*T' + RQR′   # P_{t|t-1} = Var s_{t|t-1} = T*P_{t-1|t-1}*T' + R*Q*R'
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
                 return_loglh::Bool = true, tol::AbstractFloat = 0.0) where {S<:Real}

    # Keep rows of measurement equation corresponding to non-missing observables
    nonmissing = .![ismissing(x) ? true : isnan(x) for x in y_obs]

    y_obs = y_obs[nonmissing]
    Z     = k.Z[nonmissing, :]
    D     = k.D[nonmissing]
    E     = k.E[nonmissing, nonmissing]

    Ny    = length(y_obs)

    s_pred = k.s_t
    P_pred = k.P_t

    y_pred = Z*s_pred + D             # y_{t|t-1} = Z*s_{t|t-1} + D

    V_pred     = Z*P_pred*Z' + E      # V_{t|t-1} = Var y_{t|t-1} = Z*P_{t|t-1}*Z' + E
    V_pred     = (V_pred + V_pred')/2 # V_pred should be symmetric; this guarantees symmetry and divides by 2 so entries aren't double
    V_pred_inv = inv(V_pred)
    dy         = y_obs - y_pred       # dy = y_t - y_{t|t-1} (prediction error)

    if !k.converged
        PZV = P_pred'*Z'*V_pred_inv
    end

    if size(k.PZV) == size(PZV)
        if (tol > 0.0) && (maximum(abs.(PZV - k.PZV)) < tol)
            k.converged = true
        end
    end
    k.PZV = PZV

    k.s_t = s_pred + PZV*dy       # s_{t|t} = s_{t|t-1} + P_{t|t-1}'*Z'/V_{t|t-1}*dy
    k.P_t = P_pred - PZV*Z*P_pred # P_{t|t} = P_{t|t-1} - P_{t|t-1}'*Z'/V_{t|t-1}*Z*P_{t|t-1}

    if return_loglh
        # p(y_t | y_{1:t-1})
        @show det(V_pred)
        k.loglh_t = -(Ny*log(2π) + log(det(V_pred)) + dy'*V_pred_inv*dy)/2
    end
    return nothing
end
