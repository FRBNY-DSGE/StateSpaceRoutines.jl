#=
This code is loosely based on a routine originally copyright Federal Reserve Bank of Atlanta
and written by Iskander Karibzhanov.
=#

"""
```
kalman_filter(data, TTT, RRR, CCC, QQ, ZZ, DD, EE, z0 = Vector(),
    P0 = Matrix(); likelihood_only = false, n_presample_periods = 0)

kalman_filter(regime_indices, data, TTTs, RRRs, CCCs, QQs, ZZs, DDs,
    EEs, z0 = Vector(), P0 = Matrix(); likelihood_only = false,
    n_presample_periods = 0)
```

This function implements the Kalman filter for the following state-space model:

```
z_{t+1} = CCC + TTT*z_t + RRR*ϵ_t    (transition equation)
y_t     = DD  + ZZ*z_t  + η_t        (measurement equation)

ϵ_t ∼ N(0, QQ)
η_t ∼ N(0, EE)
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

- `likelihood_only`: indicates whether we want to return only the likelihood or
  all return values
- `n_presample_periods`: if greater than 0, the first `n_presample_periods` will
  be omitted from the likelihood calculation and all return values

### Outputs

- `log_likelihood`: log likelihood of the state-space model
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
function kalman_filter{S<:AbstractFloat}(data::Matrix{S},
    TTT::Matrix{S}, RRR::Matrix{S}, CCC::Vector{S},
    QQ::Matrix{S}, ZZ::Matrix{S}, DD::Vector{S}, EE::Matrix{S},
    z0::Vector{S} = Vector{S}(), P0::Matrix{S} = Matrix{S}();
    likelihood_only::Bool = false, n_presample_periods::Int = 0)

    T = size(data, 2)
    regime_indices = Range{Int64}[1:T]

    kalman_filter(regime_indices, data, Matrix{S}[TTT], Matrix{S}[RRR], Vector{S}[CCC],
        Matrix{S}[QQ], Matrix{S}[ZZ], Vector{S}[DD], Matrix{S}[EE], z0, P0;
        likelihood_only = likelihood_only, n_presample_periods = n_presample_periods)
end

function kalman_filter{S<:AbstractFloat}(regime_indices::Vector{Range{Int64}},
    data::Matrix{S}, TTTs::Vector{Matrix{S}}, RRRs::Vector{Matrix{S}}, CCCs::Vector{Vector{S}},
    QQs::Vector{Matrix{S}}, ZZs::Vector{Matrix{S}}, DDs::Vector{Vector{S}}, EEs::Vector{Matrix{S}},
    z0::Vector{S} = Vector{S}(), P0::Matrix{S} = Matrix{S}();
    likelihood_only::Bool = false, n_presample_periods::Int = 0)

    # Dimensions
    n_regimes = length(regime_indices)
    T  = size(data,    2) # number of periods of data
    Nz = size(TTTs[1], 1) # number of states
    Ne = size(RRRs[1], 2) # number of shocks
    Ny = size(ZZs[1],  1) # number of observables

    # Populate initial conditions if they are empty
    if isempty(z0) || isempty(P0)
        e, _ = eig(TTTs[1])
        if all(abs(e) .< 1.)
            z0 = (UniformScaling(1) - TTTs[1])\CCCs[1]
            P0 = solve_discrete_lyapunov(TTTs[1], RRRs[1]*QQs[1]*RRRs[1]')
        else
            z0 = CCC
            P0 = 1e6 * eye(Nz)
        end
    end

    z = z0
    P = P0

    # Initialize outputs
    if !likelihood_only
        pred          = zeros(S, Nz, T)
        vpred         = zeros(S, Nz, Nz, T)
        yprederror    = NaN*zeros(S, Ny, T)
        ystdprederror = NaN*zeros(S, Ny, T)
        filt          = zeros(Nz, T)
        vfilt         = zeros(Nz, Nz, T)
    end

    log_likelihood = 0.0

    for i = 1:n_regimes
        # Get state-space system matrices for this regime
        regime_periods = regime_indices[i]
        regime_data = data[:, regime_periods]

        TTT, RRR, CCC = TTTs[i], RRRs[i], CCCs[i]
        ZZ,  DD       = ZZs[i],  DDs[i]
        QQ,  EE       = QQs[i],  EEs[i]

        V = RRR*QQ*RRR'    # V = Var(z_t) = Var(Rϵ_t)

        for t in regime_periods
            # Index out rows of the measurement equation for which we have
            # nonmissing data in period t
            nonmissing = !isnan(data[:, t])
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

            if !likelihood_only
                pred[:, t]                   = z
                vpred[:, :, t]               = P
                yprederror[nonmissing, t]    = dy
                ystdprederror[nonmissing, t] = dy ./ sqrt(diag(V))
            end

            ## Compute marginal log-likelihood, log P(y_t|y_1,...y_{t-1},θ)
            ## log P(y_1,...,y_T|θ) ∝ log P(y_1|θ) + log P(y_2|y_1,θ) + ... + P(y_T|y_1,...,y_{T-1},θ)
            if t > n_presample_periods
                log_likelihood += -log(det(V))/2 - first(dy'*ddy/2) - Ny_t*log(2*pi)/2
            end

            ## Update
            z = z + P'*ZZ_t'*ddy            # z_{t|t} = z_{t|t-1} + P_{t|t-1}'*ZZ'*(1/V_{t|t-1})dy
            P = P - P'*ZZ_t'/V*ZZ_t*P       # P_{t|t} = P_{t|t-1} - P_{t|t-1}'*ZZ'*(1/V_{t|t-1})*ZZ*P_{t|t-1}

            if !likelihood_only
                filt[:, t]     = z
                vfilt[:, :, t] = P
            end

        end # of loop through this regime's periods

    end # of loop through regimes

    if !likelihood_only && n_presample_periods > 0
        mainsample_periods = n_presample_periods+1:T

        # If we choose to discard presample periods, then we reassign `z0`
        # and `P0` to be their values at the end of the presample/beginning
        # of the main sample
        z0 = squeeze(filt[:,     n_presample_periods], 2)
        P0 = squeeze(vfilt[:, :, n_presample_periods], 3)

        pred          = pred[:,     mainsample_periods]
        vpred         = vpred[:, :, mainsample_periods]
        filt          = filt[:,     mainsample_periods]
        vfilt         = vfilt[:, :, mainsample_periods]
        yprederror    = yprederror[:,       mainsample_periods]
        ystdprederror = ypredstderror[:, :, mainsample_periods]
    end

    if !likelihood_only
        rmse = sqrt(mean((yprederror.^2)', 1))
        rmsd = sqrt(mean((ystdprederror.^2)', 1))

        return log_likelihood, pred, vpred, filt, vfilt, yprederror, ystdprederror, rmse, rmsd, z0, P0
    else
        return log_likelihood
    end
end