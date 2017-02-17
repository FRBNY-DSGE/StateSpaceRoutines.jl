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