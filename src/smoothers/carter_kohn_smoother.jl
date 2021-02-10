"""
```
carter_kohn_smoother(y, T, R, C, Q, Z, D, E, s_0, P_0;
    Nt0 = 0, draw_states = true)

carter_kohn_smoother(regime_indices, y, Ts, Rs, Cs, Qs,
    Zs, Ds, Es, s_0, P_0; Nt0 = 0, draw_states = true)
```
This program is a simulation smoother based on Carter and Kohn's \"On Gibbs
Sampling for State Space Models\" (Biometrika, 1994). It recursively samples
from the conditional distribution of time-t states given the full set of
observables and states from time t+1 to time T. Unlike the Durbin-Koopman
simulation smoother, this one does rely on inverting potentially singular
matrices using the Moore-Penrose pseudoinverse.

The state space is augmented with shocks (see `?augment_states_with_shocks`),
and the augmented state space is Kalman filtered and smoothed. Finally, the
smoothed states and shocks are indexed out of the augmented state vectors.

The original state space (before augmenting with shocks) is given by:

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

**Method 1 only:** state-space system matrices `T`, `R`, `C`, `Q`, `Z`,
`D`, `E`. See `?kalman_filter`
pp
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
  `N(s_{t|T}, P_{t|T})` or to use the mean `s_{t|T}` (reducing to the Hamilton
  smoother)

### Outputs

- `s_smth`: `Ns` x `Nt` matrix of smoothed states `s_{t|T}`
- `ϵ_smth`: `Ne` x `Nt` matrix of smoothed shocks `ϵ_{t|T}`
"""
function carter_kohn_smoother(y::AbstractMatrix,
                              T::Matrix{S}, R::Matrix{S}, C::Vector{S},
                              Q::Matrix{S}, Z::Matrix{S}, D::Vector{S}, E::Matrix{S},
                              s_0::Vector{S}, P_0::Matrix{S};
                              Nt0::Int = 0, draw_states::Bool = true,
                              testing_carter_kohn::Bool = false) where {S<:AbstractFloat}

    Nt = size(y, 2)
    carter_kohn_smoother(UnitRange{Int}[1:Nt], y, Matrix{S}[T], Matrix{S}[R], Vector{S}[C],
                         Matrix{S}[Q], Matrix{S}[Z], Vector{S}[D], Matrix{S}[E], s_0, P_0;
                         Nt0 = Nt0, draw_states = draw_states,
                         testing_carter_kohn = testing_carter_kohn)
end

function carter_kohn_smoother(regime_indices::Vector{UnitRange{Int}}, y::AbstractMatrix,
    Ts::Vector{Matrix{S}}, Rs::Vector{Matrix{S}}, Cs::Vector{Vector{S}}, Qs::Vector{Matrix{S}},
    Zs::Vector{Matrix{S}}, Ds::Vector{Vector{S}}, Es::Vector{Matrix{S}},
    s_0::Vector{S}, P_0::Matrix{S};
    Nt0::Int = 0, draw_states::Bool = true, testing_carter_kohn::Bool = false) where {S<:AbstractFloat}

    # Dimensions
    Nt = size(y,     2) # number of periods of data
    Ns = size(Ts[1], 1) # number of states
    Ne = size(Rs[1], 2) # number of shocks

    # Augment state space with shocks
    Ts, Rs, Cs, Zs, s_0, P_0 =
        augment_states_with_shocks(regime_indices, Ts, Rs, Cs, Qs, Zs, s_0, P_0)

    # Kalman filter stacked states and shocks stil_t
    _, stil_pred, Ptil_pred, stil_filt, Ptil_filt, _, _, _, _ =
        kalman_filter(regime_indices, y, Ts, Rs, Cs, Qs, Zs, Ds, Es, s_0, P_0,
                      outputs = [:pred, :filt])

    # Smooth the stacked states and shocks recursively, starting at t = T-1 and
    # going backwards
    stil_smth = copy(stil_filt)

    if testing_carter_kohn
        conded = zeros(regime_indices[end][end])
    end

    for i = length(regime_indices):-1:1
        # Get state-space system matrices for this regime
        T = Ts[i]

        # Perform within-regime smoothing recursion
        reg_end_index = regime_indices[i][end] # avoid indexing into this value every time during the loop to save on time
        for t in reverse(regime_indices[i])
            if t == Nt
                μ = @view stil_filt[:, end]
                Σ = @view Ptil_filt[:, :, end]
            else
                # Need to be careful in calculating J b/c the T matrix required is supposed to be T_{t + 1}.
                # If the T is time-varying, then we need to be careful in the time period right before
                # a regime switch. For notes showing that this is true, see
                # https://christophertonetti.com/files/notes/Nakata_Tonetti_KalmanFilterAndSmoother.pdf,
                # which shows
                # J_t = P_{t | t} * T_{t + 1} * P⁻¹_{t + 1 | t}
                correct_T = t == reg_end_index ? Ts[i + 1] : T
                J = view(Ptil_filt, :, :, t) * correct_T' * pinv(view(Ptil_pred, :, :, t + 1))
                μ = view(stil_filt, :, t) + J * (view(stil_smth, :, t + 1) - view(stil_pred, :, t + 1)) # stil_{t|T}
                Σ = view(Ptil_filt, :, :, t) - J * correct_T * view(Ptil_filt, :, :, t)                 # Ptil_{t|T}
            end

            # Draw stil_t ∼ N(stil_{t|T}, Ptil_{t|T})
            stil_smth[:, t] = if draw_states
                U, eig, _ = svd(Σ)
                if testing_carter_kohn
                    U1, eig1, _ = svd(Σ[1:Ns, 1:Ns])
                    U2, eig2, _ = svd(Σ[Ns+1:end, Ns+1:end])
                    conded[t] = maximum(abs.(Σ[Ns+1:end,1:Ns]))
                    # conded[t] = isposdef(Σ[Ns+1:end,Ns+1:end]) ? 1.0 : 0.0
                    # conded[t] = minimum(eig2)
                    # conded[t] = isposdef(Σ[1:Ns,1:Ns]) ? 1.0 : 0.0
                    # conded[t] = maximum(abs.(U[Ns+1:end,:]))
                    # conded[t] = maximum(abs.((U * diagm(0 => (sqrt.(eig))))[Ns+1:end,:]))
                    # conded[t] = minimum(U[Ns+1:end,:])
                    # conded[t] = maximum(abs.(randn(Ns+Ne)))
                    # @show size(U), size(eig), size(Σ), size(μ)
                    # firsted = μ[1:Ns] .+ U[1:Ns,1:Ns] * diagm(0 => (sqrt.(eig[1:Ns]))) * randn(Ns)
                    # seconded = μ[Ns+1:end] .+ diagm(0 => (sqrt.(eig[Ns+1:end]))) * randn(Ne)# Σ[Ns+1:end, Ns+1:end] * randn(Ne)
                    # vcat(firsted, seconded)
                    # evals, evecs = eigen(Σ)
                    # please_work = evecs * diagm(0 => sqrt.(evals)) * inv(evecs)
                    # real(μ .+ please_work * randn(Ns+Ne))
                    # μ .+ Σ * randn(Ns+Ne)
                    vcat(μ[1:Ns] .+ U1 * diagm(0 => sqrt.(eig1)) * randn(Ns),
                         μ[Ns+1:end] .+ diagm(0 => sqrt.(eig2)) * randn(Ne))
                else
                    μ .+ U * Diagonal(sqrt.(eig)) * randn(Ns+Ne)
                end
            else
                μ
            end
        end
    end

    # Index out states and shocks
    s_smth = stil_smth[1:Ns,     :] # s_{t|T}
    ϵ_smth = stil_smth[Ns+1:end, :] # ϵ_{t|T}

    # Trim the presample if needed
    if Nt0 > 0
        insample = Nt0+1:Nt
        s_smth = s_smth[:, insample]
        ϵ_smth = ϵ_smth[:, insample]
    end

    if testing_carter_kohn
        return s_smth, ϵ_smth, conded
    end

    return s_smth, ϵ_smth
end
