"""
```
augment_states_with_shocks(regime_indices, TTTs, RRRs, CCCs, QQs, ZZs, z0, P0)
```

Go from original state space:

```
z_{t+1} = CCC + TTT*z_t + RRR*ϵ_t    (transition equation)
y_t     = DD  + ZZ*z_t  + η_t        (measurement equation)

ϵ_t ∼ N(0, QQ)
η_t ∼ N(0, EE)
Cov(ϵ_t, η_t) = 0
```

to augmented state space:

```
|z_{t+1}| = |CCC| + |TTT 0| |z_t| + |RRR| ϵ_t    (transition equation)
|ϵ_{t+1}|   | 0 |   | 0  0| |ϵ_t|   | I |

y_t = DD + |ZZ| |z_t| + η_t                      (measurement equation)
           | 0| |ϵ_t|

ϵ_t ∼ N(0, QQ)
η_t ∼ N(0, EE)
Cov(ϵ_t, η_t) = 0
```

with initial state and covariance:

```
|z0| and |P0  0|
| 0|     | 0 QQ|
```

Returns the augmented `TTTs`, `RRRs`, CCCs`, `ZZs`, `z0`, and `P0`.
"""
function augment_states_with_shocks(regime_indices::Vector{AbstractRange{Int64}},
    TTTs::Vector{Matrix{S}}, RRRs::Vector{Matrix{S}}, CCCs::Vector{Vector{S}},
    QQs::Vector{Matrix{S}}, ZZs::Vector{Matrix{S}}, z0::Vector{S}, P0::Matrix{S}) where S <: AbstractFloat

    n_regimes = length(regime_indices)

    # Dimensions
    Nz = size(TTTs[1], 1) # number of states
    Ne = size(RRRs[1], 2) # number of shocks
    Ny = size(ZZs[1],  1) # number of observables

    # Augment initial conditions
    z0 = vcat(z0, zeros(Ne))
    P0 = vcat(hcat(P0, zeros(Nz, Ne)), hcat(zeros(Ne, Nz), QQs[1]))

    for i = 1:n_regimes
        # Get state-space system matrices for this regime
        TTT, RRR, CCC = TTTs[i], RRRs[i], CCCs[i]
        QQ,  ZZ       = QQs[i],  ZZs[i]

        # Augment regime-specific matrices
        TTTs[i] = vcat(hcat(TTT, zeros(Nz, Ne)), hcat(zeros(Ne, Nz), zeros(Ne, Ne)))
        RRRs[i] = vcat(RRR, Matrix{Float64}(I, Ne, Ne))
        CCCs[i] = vcat(CCC, zeros(Ne))
        ZZs[i]  = hcat(ZZ, zeros(Ny, Ne))
    end

    return TTTs, RRRs, CCCs, ZZs, z0, P0
end