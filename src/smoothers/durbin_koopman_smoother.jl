"""
```
durbin_koopman_smoother(y, T, R, C, Q, Z, D, E, s_0, P_0;
    Nt0 = 0, draw_states = true)

durbin_koopman_smoother(regime_indices, y, Ts, Rs, Cs, Qs,
    Zs, Ds, Es, s_0, P_0; Nt0 = 0, draw_states = true)
```

This program is a simulation smoother based on Durbin and Koopman's
\"A Simple and Efficient Simulation Smoother for State Space Time Series
Analysis\" (Biometrika, 2002).

Unlike other simulation smoothers (for example, that of Carter and Kohn,
1994), this method does not require separate draws for each period, draws
of the state vectors, or even draws from a conditional distribution.
Instead, vectors of shocks are drawn from the unconditional distribution
of shocks, which is then corrected (via a Kalman smoothing step), to
yield a draw of shocks conditional on the data. This is then used to
generate a draw of states conditional on the data. Drawing the states in
this way is much more efficient than other methods, as it avoids the need
for multiple draws of state vectors (requiring singular value
decompositions), as well as inverting state covariance matrices
(requiring the use of the computationally intensive and relatively
erratic Moore-Penrose pseudoinverse).

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
- `pred`: `Ns` x `Nt` matrix of one-step predicted state vectors `s_{t|t-1}`
  (from the Kalman filter)
- `vpred`: `Ns` x `Ns` x `Nt` array of mean squared errors `P_{t|t-1}` of
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

- `Nt0`: if greater than 0, the returned smoothed states and shocks matrices
  will be shorter by that number of columns (taken from the beginning)
- `draw_states`: indicates whether to draw smoothed states from the distribution
  `N(s_{t|T}, P_{t|T})` or to use the mean `s_{t|T}` (reducing to the Koopman
  smoother)

### Outputs

- `s_smth`: `Ns` x `Nt` matrix of smoothed states `s_{t|T}`
- `ϵ_smth`: `Ne` x `Nt` matrix of smoothed shocks `ϵ_{t|T}`
"""
function durbin_koopman_smoother(y::AbstractArray{Union{S, Missing}},
    T::Matrix{S}, R::Matrix{S}, C::Vector{S},
    Q::Matrix{S}, Z::Matrix{S}, D::Vector{S}, E::Matrix{S},
    s_0::Vector{S}, P_0::Matrix{S};
    Nt0::Int = 0, draw_states::Bool = true) where {S<:AbstractFloat}

    Nt = size(y, 2)
    durbin_koopman_smoother(AbstractRange{Int}[1:Nt], y, Matrix{S}[T], Matrix{S}[R], Vector{S}[C],
        Matrix{S}[Q], Matrix{S}[Z], Vector{S}[D], Matrix{S}[E], s_0, P_0;
        Nt0 = Nt0, draw_states = draw_states)
end

function durbin_koopman_smoother(regime_indices::Vector{AbstractRange{Int}}, y::AbstractArray{Union{S, Missing}},
    Ts::Vector{Matrix{S}}, Rs::Vector{Matrix{S}}, Cs::Vector{Vector{S}}, Qs::Vector{Matrix{S}},
    Zs::Vector{Matrix{S}}, Ds::Vector{Vector{S}}, Es::Vector{Matrix{S}},
    s_0::Vector{S}, P_0::Matrix{S};
    Nt0::Int = 0, draw_states::Bool = true) where {S<:AbstractFloat}

    # Dimensions
    n_regimes = length(regime_indices)
    Nt = size(y,     2) # number of periods of data
    Ns = size(Ts[1], 1) # number of states
    Ne = size(Rs[1], 2) # number of shocks
    Ny = size(Zs[1], 1) # number of observables

    # Draw initial state s_0+
    s_plus_t = if draw_states
        U, eig, _ = svd(P_0)
        U * diagm(0 => (sqrt.(eig))) * randn(Ns)
    else
        zeros(S, Ns)
    end

    # Produce "fake" states and observables (s+ and y+) by
    # iterating the state-space system forward, drawing shocks ϵ+
    s_plus = zeros(S, Ns, Nt)
    ϵ_plus = zeros(S, Ne, Nt)
    y_plus = zeros(Union{S, Missing}, Ny, Nt)

    for i = 1:n_regimes
        # Get state-space system matrices for this regime
        T, R, C, Q = Ts[i], Rs[i], Cs[i], Qs[i]
        Z, D       = Zs[i], Ds[i]
        for t in regime_indices[i]
            s_plus_t = if draw_states
                ϵ_plus[:, t] = sqrt.(Qs[i]) * randn(Ne)
                T*s_plus_t + R*ϵ_plus[:, t] + C
            else
                T*s_plus_t + C
            end
            s_plus[:, t] = s_plus_t
            y_plus[:, t] = Z*s_plus_t + D
        end
    end

    # Replace y+ with NaNs wherever y has NaNs
    y_plus[ismissing.(y)] .= missing

    # Compute y* = y - y+
    y_star = y .- y_plus

    # Cast to Matrix{Union{S, Missing}} to ensure
    # conformity because for some reason
    # arithmetic operators on two Matrix{Union{S, Missing}} returns
    # a matrix of concrete type S.
    y_star = convert(Matrix{Union{S, Missing}}, y_star)

    # Run the Kalman filter on y*
    # Note that we pass in `zeros(Ny)` instead of `D` because the
    # measurement equation for y* has no constant term
    Ds_star = fill(zeros(Ny), n_regimes)
    _, s_pred, P_pred, _, _, _, _, _, _ = kalman_filter(regime_indices, y_star, Ts, Rs, Cs, Qs, Zs, Ds_star, Es, s_0, P_0; outputs = [:pred])

    # Kalman smooth y*
    s_star, ϵ_star = koopman_smoother(regime_indices, y_star, Ts, Rs, Cs, Qs,
                                      Zs, Ds_star, Es, s_0, P_0, s_pred, P_pred)

    # Compute smoothed states and shocks
    s_smth = s_plus + s_star
    ϵ_smth = ϵ_plus + ϵ_star

    # Trim the presample if needed
    if Nt0 > 0
        insample = Nt0+1:Nt

        s_smth = s_smth[:, insample]
        ϵ_smth = ϵ_smth[:, insample]
    end
    return s_smth, ϵ_smth
end
