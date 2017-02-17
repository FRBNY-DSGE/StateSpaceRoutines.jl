function solve_smoothed_shocks{S<:AbstractFloat}(regime_indices::Vector{Range{Int64}},
    TTTs::Vector{Matrix{S}}, RRRs::Vector{Matrix{S}},
    z0::Vector{S}, smoothed_states::Matrix{S})

    n_regimes = length(regime_indices)

    # Dimensions
    T  = size(smoothed_states, 2) # number of periods of data
    Nz = size(TTTs[1], 1) # number of states
    Ne = size(RRRs[1], 2) # number of shocks

    # Solve for shocks needed to produce smoothed states in each period
    smoothed_shocks = zeros(Ne, T)

    for i = 1:n_regimes
        # Get state-space system matrices for this regime
        regime_periods = regime_indices[i]
        TTT, RRR = TTTs[i], RRRs[i]

        if rank(RRR) < Ne
            warn("RRR is not sufficient rank to map forecast errors uniquely onto shocks")
        end
        RRR_inv = pinv(RRR)

        for t in regime_periods
            z_t = smoothed_states[:, t]
            z_t1 = (t == 1) ? z0 : smoothed_states[:, t-1]
            smoothed_shocks[:, t] = RRR_inv*(z_t - TTT*z_t1)
        end
    end

    return smoothed_shocks
end

function augment_states_with_shocks{S<:AbstractFloat}(regime_indices::Vector{Range{Int64}},
    TTTs::Vector{Matrix{S}}, RRRs::Vector{Matrix{S}}, CCCs::Vector{Vector{S}},
    QQs::Vector{Matrix{S}}, ZZs::Vector{Matrix{S}},
    z0::Vector{S} = Vector{S}(), P0::Matrix{S} = Matrix{S}())

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
        RRRs[i] = vcat(RRR, eye(Ne))
        CCCs[i] = vcat(CCC, zeros(Ne))
        ZZs[i]  = hcat(ZZ, zeros(Ny, Ne))
    end

    return TTTs, RRRs, CCCs, ZZs, z0, P0
end