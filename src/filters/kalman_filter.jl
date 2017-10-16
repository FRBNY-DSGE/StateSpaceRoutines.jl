#=
This code is loosely based on a routine originally copyright Federal Reserve Bank of Atlanta
and written by Iskander Karibzhanov.
=#

"""
```
kalman_filter(data, TTT, RRR, CCC, QQ, ZZ, DD, EE, z0 = Vector(), P0 = Matrix();
    allout = true, n_presample_periods = 0)

kalman_filter(regime_indices, data, TTTs, RRRs, CCCs, QQs, ZZs, DDs,
    EEs, z0 = Vector(), P0 = Matrix(); allout = true, n_presample_periods = 0)
```

This function implements the Kalman filter for the following state-space model:

```
z_{t+1} = CCC + TTT*z_t + RRR*ϵ_t    (transition equation)
y_t     = DD  + ZZ*z_t  + η_t        (measurement equation)

ϵ_t ∼ N(0, QQ)
η_t ∼ N(0, EE)
Cov(ϵ_t, η_t) = 0
```

### Inputs

- `data`: `Ny` x `T` matrix containing data `y(1), ... , y(T)`
- `z0`: optional `Nz` x 1 initial state vector
- `P0`: optional `Nz` x `Nz` initial state covariance matrix

**Method 1 only:**

- `TTT`: `Nz` x `Nz` state transition matrix
- `RRR`: `Nz` x `Ne` matrix in the transition equation mapping shocks to states
- `CCC`: `Nz` x 1 constant vector in the transition equation
- `QQ`: `Ne` x `Ne` matrix of shock covariances
- `ZZ`: `Ny` x `Nz` matrix in the measurement equation mapping states to
  observables
- `DD`: `Ny` x 1 constant vector in the measurement equation
- `EE`: `Ny` x `Ny` matrix of measurement error covariances

**Method 2 only:**

- `regime_indices`: `Vector{Range{Int64}}` of length `n_regimes`, where
  `regime_indices[i]` indicates the time periods `t` in regime `i`
- `TTTs`: `Vector{Matrix{S}}` of `TTT` matrices for each regime
- `RRRs`
- `CCCs`
- `QQs`
- `ZZs`
- `DDs`
- `EEs`

where:

- `T`: number of time periods for which we have data
- `Nz`: number of states
- `Ne`: number of shocks
- `Ny`: number of observables

### Keyword Arguments

- `allout`: indicates whether we want to return all values. If `!allout`, then
  we return only the likelihood, `z_{T|T}`, and `P_{T|T}`.
- `n_presample_periods`: if greater than 0, the first `n_presample_periods` will
  be omitted from the likelihood calculation and all return values

### Outputs

- `log_likelihood`: log likelihood of the state-space model
- `zend`: `Nz` x 1 final filtered state `z_{T|T}`
- `Pend`: `Nz` x `Nz` final filtered state covariance matrix `P_{T|T}`
- `pred`: `Nz` x `T` matrix of one-step predicted state vectors `z_{t|t-1}`
- `vpred`: `Nz` x `Nz` x `T` array of mean squared errors `P_{t|t-1}` of
  predicted state vectors
- `filt`: `Nz` x `T` matrix of filtered state vectors `z_{t|t}`
- `vfilt`: `Nz` x `Nz` x `T` matrix containing mean squared errors `P_{t|t}` of
  filtered state vectors
- `yprederror`: `Ny` x `T` matrix of observable prediction errors
  `y_t - y_{t|t-1}`
- `ystdprederror`: `Ny` x `T` matrix of standardized observable prediction errors
  `V_{t|t-1} \ (y_t - y_{t|t-1})`, where `y_t - y_{t|t-1} ∼ N(0, V_{t|t-1}`
- `rmse`: 1 x `T` row vector of root mean squared prediction errors
- `rmsd`: 1 x `T` row vector of root mean squared standardized prediction errors
- `z0`: `Nz` x 1 initial state vector. This may have reassigned to the last
  presample state vector if `n_presample_periods > 0`
- `P0`: `Nz` x `Nz` initial state covariance matrix. This may have reassigned to
  the last presample state covariance if `n_presample_periods > 0`
- `marginal_loglh`: a vector of the marginal log likelihoods from t = 1 to T

### Notes

When `z0` and `P0` are omitted, the initial state vector and its covariance
matrix of the time invariant Kalman filters are computed under the stationarity
condition:

```
z0  = (I - TTT)\CCC
P0 = reshape(I - kron(TTT, TTT))\vec(RRR*QQ*RRR'), Nz, Nz)
```

where:

- `kron(TTT, TTT)` is a matrix of dimension `Nz^2` x `Nz^2`, the Kronecker
  product of `TTT`
- `vec(RRR*QQ*RRR')` is the `Nz^2` x 1 column vector constructed by stacking the
  `Nz` columns of `RRR*QQ*RRR'`

All eigenvalues of `TTT` are inside the unit circle when the state space model
is stationary.  When the preceding formula cannot be applied, the initial state
vector estimate is set to `CCC` and its covariance matrix is given by `1e6 * I`.
"""
function kalman_filter{S<:AbstractFloat}(regime_indices::Vector{Range{Int64}},
    data::Matrix{S}, TTTs::Vector{Matrix{S}}, RRRs::Vector{Matrix{S}}, CCCs::Vector{Vector{S}},
    QQs::Vector{Matrix{S}}, ZZs::Vector{Matrix{S}}, DDs::Vector{Vector{S}}, EEs::Vector{Matrix{S}},
    z0::Vector{S} = Vector{S}(), P0::Matrix{S} = Matrix{S}();
    allout::Bool = true, n_presample_periods::Int = 0)

    # Dimensions
    T  = size(data,    2) # number of periods of data
    Nz = size(TTTs[1], 1) # number of states
    Ny = size(ZZs[1],  1) # number of observables

    # Initialize outputs
    z = z0
    P = P0
    log_likelihood = zero(S)
    if allout
        pred  = zeros(S, Nz, T)
        vpred = zeros(S, Nz, Nz, T)
        filt  = zeros(S, Nz, T)
        vfilt = zeros(S, Nz, Nz, T)
        yprederror    = zeros(S, Ny, T)
        ystdprederror = zeros(S, Ny, T)
        rmse  = zeros(S, 1, Ny)
        rmsd  = zeros(S, 1, Ny)
        marginal_loglh = zeros(T - n_presample_periods)
    end

    # Iterate through regimes
    for i = 1:length(regime_indices)
        ts = regime_indices[i]
        regime_data = data[:, ts]
        T0 = i == 1 ? n_presample_periods : 0

        if allout
            L, z, P, pred[:,ts], vpred[:,:,ts], filt[:,ts], vfilt[:,:,ts],
            yprederror[:,ts], ystdprederror[:,ts], _, _, z0_, P0_, marginal_loglh[ts] =
                kalman_filter(regime_data, TTTs[i], RRRs[i], CCCs[i], QQs[i], ZZs[i], DDs[i], EEs[i], z, P;
                              allout = true, n_presample_periods = T0)

            # If `n_presample_periods > 0`, then `z0_` and `P0_` are returned as
            # the filtered values at the end of the presample/beginning of the
            # main sample (i.e. not the same the `z0` and `P0` passed into this
            # method, which are from the beginning of the presample). If we are
            # in the first regime, we want to reassign `z0` and `P0`
            # accordingly.
            if i == 1
                z0, P0 = z0_, P0_
            end
        else
            L, z, P = kalman_filter(regime_data, TTTs[i], RRRs[i], CCCs[i], QQs[i], ZZs[i], DDs[i], EEs[i], z, P,
                                    allout = false, n_presample_periods = T0)
        end
        log_likelihood += L
    end

    if allout
        rmse = sqrt.(mean((yprederror.^2)', 1))
        rmsd = sqrt.(mean((ystdprederror.^2)', 1))

        return log_likelihood, z, P, pred, vpred, filt, vfilt, yprederror, ystdprederror, rmse, rmsd, z0, P0,
        marginal_loglh
    else
        return log_likelihood, z, P
    end
end

function kalman_filter{S<:AbstractFloat}(data::Matrix{S},
    TTT::Matrix{S}, RRR::Matrix{S}, CCC::Vector{S},
    QQ::Matrix{S}, ZZ::Matrix{S}, DD::Vector{S}, EE::Matrix{S},
    z0::Vector{S} = Vector{S}(), P0::Matrix{S} = Matrix{S}(0,0);
    allout::Bool = true, n_presample_periods::Int = 0)

    # Dimensions
    T  = size(data, 2) # number of periods of data
    Nz = size(TTT,  1) # number of states
    Ne = size(RRR,  2) # number of shocks
    Ny = size(ZZ,   1) # number of observables

    # Populate initial conditions if they are empty
    if isempty(z0) || isempty(P0)
        e, _ = eig(TTT)
        if all(abs.(e) .< 1.)
            z0 = (UniformScaling(1) - TTT)\CCC
            P0 = solve_discrete_lyapunov(TTT, RRR*QQ*RRR')
        else
            z0 = CCC
            P0 = 1e6 * eye(Nz)
        end
    end

    z = z0
    P = P0

    # Initialize outputs
    marginal_loglh = zeros(T)
    if allout
        pred                = zeros(S, Nz, T)
        vpred               = zeros(S, Nz, Nz, T)
        filt                = zeros(S, Nz, T)
        vfilt               = zeros(S, Nz, Nz, T)
        yprederror          = NaN*zeros(S, Ny, T)
        ystdprederror       = NaN*zeros(S, Ny, T)
    end

    V = RRR*QQ*RRR' # V = Var(z_t) = Var(Rϵ_t)

    for t = 1:T
        # Index out rows of the measurement equation for which we have
        # nonmissing data in period t
        nonmissing = .!isnan.(data[:, t])
        y_t  = data[nonmissing, t]
        ZZ_t = ZZ[nonmissing, :]
        DD_t = DD[nonmissing]
        EE_t = EE[nonmissing, nonmissing]
        Ny_t = length(y_t)

        ## Forecast
        z = TTT*z + CCC                 # z_{t|t-1} = TTT*z_{t-1|t-1} + CCC
        P = TTT*P*TTT' + RRR*QQ*RRR'    # P_{t|t-1} = Var s_{t|t-1} = TTT*P_{t-1|t-1}*TTT' + RRR*QQ*RRR'
        V = ZZ_t*P*ZZ_t' + EE_t         # V_{t|t-1} = Var y_{t|t-1} = ZZ*P_{t|t-1}*ZZ' + EE
        V = (V+V')/2

        dy = y_t - ZZ_t*z - DD_t        # dy  = y_t - y_{t|t-1} = prediction error
        ddy = V\dy                      # ddy = (1/V_{t|t-1})dy = weighted prediction error

        if allout
            pred[:, t]                   = z
            vpred[:, :, t]               = P
            yprederror[nonmissing, t]    = dy
            ystdprederror[nonmissing, t] = dy ./ sqrt.(diag(V))
        end

        ## Compute marginal log-likelihood, log P(y_t|y_1,...y_{t-1},θ)
        ## log P(y_1,...,y_T|θ) ∝ log P(y_1|θ) + log P(y_2|y_1,θ) + ... + P(y_T|y_1,...,y_{T-1},θ)
        if t > n_presample_periods
            marginal_loglh[t] = -log(det(V))/2 - first(dy'*ddy/2) - Ny_t*log(2*pi)/2
        end

        ## Update
        z = z + P'*ZZ_t'*ddy            # z_{t|t} = z_{t|t-1} + P_{t|t-1}'*ZZ'*(1/V_{t|t-1})dy
        P = P - P'*ZZ_t'/V*ZZ_t*P       # P_{t|t} = P_{t|t-1} - P_{t|t-1}'*ZZ'*(1/V_{t|t-1})*ZZ*P_{t|t-1}

        if allout
            filt[:, t]     = z
            vfilt[:, :, t] = P
        end

    end # of loop through periods

    if n_presample_periods > 0
        mainsample_periods = n_presample_periods+1:T

        marginal_loglh = marginal_loglh[mainsample_periods]

        if allout
            # If we choose to discard presample periods, then we reassign `z0`
            # and `P0` to be their values at the end of the presample/beginning
            # of the main sample
            z0 = filt[:,     n_presample_periods]
            P0 = vfilt[:, :, n_presample_periods]

            pred            = pred[:,     mainsample_periods]
            vpred           = vpred[:, :, mainsample_periods]
            filt            = filt[:,     mainsample_periods]
            vfilt           = vfilt[:, :, mainsample_periods]
            yprederror      = yprederror[:,  mainsample_periods]
            ystdprederror   = ystdprederror[:, mainsample_periods]
        end
    end

    log_likelihood = sum(marginal_loglh)

    if allout
        rmse = sqrt.(mean((yprederror.^2)', 1))
        rmsd = sqrt.(mean((ystdprederror.^2)', 1))

        return log_likelihood, z, P, pred, vpred, filt, vfilt, yprederror, ystdprederror,
        rmse, rmsd, z0, P0, marginal_loglh
    else
        return log_likelihood, z, P
    end
end
