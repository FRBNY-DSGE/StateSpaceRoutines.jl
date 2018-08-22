# s_t = TT*s_{t-1} + RR*ε_t, ε_t ~ iidN(0, QQ)
# y_t = DD+ ZZ*s_t + η_t, η_t ~ iidN(0, HH)
#y is nobs x ny with each row being time period

function chand_recursion(y::Matrix{S},
                         T::Matrix{S}, R::Matrix{S}, C::Vector{S},
                         Q::Matrix{S}, Z::Matrix{S}, D::Vector{S}, H::Matrix{S},
                         s_t::Vector{S} = Vector{S}(0), P_0::Matrix{S} = Matrix{S}(0,0);
                         allout::Bool = true, Nt0::Int = 0) where {S<:AbstractFloat}
    # Dimensions
    Ns = size(T, 1) # number of states
    Ny, Nt = size(y) # number of periods of data

    # Initialize s_t and P_0
    if isempty(s_t) || isempty(P_0)
        s_t, P_0 = init_stationary_states(T, R, C, Q)
    end

    # V_{t|t-1} = Var(y_t|y_{1:t-1}) (numerically stable)
    V_t = Z*P_0*Z' + H
    V_t = 0.5*(V_t+V_t')
    invV_t = inv(V_t)

    # We write ΔP in terms of W_t and M_t           ΔP = W_t * M_t * W_t'
    W_t = T*P_0*Z' # (Eq 12)
    M_t = -invV_t     # (Eq 13)
    kal_gain = W_t*invV_t

    # Initialize loglikelihoods to zeros so that the ones we don't update (those in presample) contribute zero towards sum of loglikelihoods.
    loglh = zeros(Nt)

    for t in 1:Nt
        # Step 1: Compute forecast error, ν_t and evaluate likelihoood
        yhat = Z*s_t + D
        ν_t = y[:, t] - yhat

        #if t > Nt0
        loglh[t] = logpdf(MvNormal(zeros(Ny), V_t), ν_t)
        #end

        # Step 2: Compute s_{t+1} using Eq. 5
        s_t = T*s_t + kal_gain*ν_t

        # Intermediate calculations to re-use
        ZW_t = Z*W_t
        MWpZp = M_t*(ZW_t')
        TW_t = T*W_t

        # Step 3: Update forecast error variance F_{t+1} (Eq 19)
        V_t1  = V_t + ZW_t*MWpZp        # F_{t+1}
        V_t1  = 0.5*(V_t1+V_t1')
        invV_t1 = inv(V_t1)

        # Step 4: Update Kalman Gain (Eq 20). Recall that kalgain = K_t * V_t⁻¹
        # Kalgain_{t+1} = (Kalgain_t*V_t + T*W_t*M_t*W_t'*Z')V_t⁻¹
        kal_gain = (kal_gain*V_t + TW_t*MWpZp)*invV_t1

        # Step 5: Update W
        W_t = TW_t - kal_gain*Z*W_t    # W_{t+1}

        # Step 6: Update M
        M_t = M_t + MWpZp*invV_t*MWpZp'     # M_{t+1}
        M_t = 0.5*(M_t + M_t')

        # Finish updates of forecast error variances
        V_t = V_t1
        invV_t = invV_t1
    end
    loglh, _, _, _, _ = remove_presample!(Nt0, loglh, zeros(0,0), zeros(0,0,0), zeros(0,0), zeros(0,0,0))
    if allout
        return sum(loglh), loglh
    else
        return loglh
    end
end

function chand_recursion(regime_indices::Vector{Range{Int}}, y::Matrix{S},
                         Ts::Vector{Matrix{S}}, Rs::Vector{Matrix{S}}, Cs::Vector{Vector{S}},
                         Qs::Vector{Matrix{S}}, Zs::Vector{Matrix{S}}, Ds::Vector{Vector{S}},
                         Es::Vector{Matrix{S}},
                         s_0::Vector{S} = Vector{S}(0), P_0::Matrix{S} = Matrix{S}(0,0);
                         outputs::Vector{Symbol} = [:loglh, :pred, :filt],
                         Nt0::Int = 0) where {S<:AbstractFloat}
    Ns = size(Ts[1], 1)
    Nt = size(y, 2)
    @assert first(regime_indices[1]) == 1
    @assert last(regime_indices[end]) == Nt

    loglh = Vector{S}(0)

    for i=1:length(regime_indices)
        ts = regime_indices[i]
        loglh, loglh_i = chand_recursion(y[:ts], Ts[i], Rs[i], Cs[i], Qs[i], Zs[i], Ds[i], Es[i],
                                         s_t, P_t; allout = true, Nt0 = 0)
        loglh[ts] = loglh_i
    end
    loglh, _, _, _, _ = remove_presample!(Nt0, loglh, zeros(0), zeros(0), zeros(0), zeros(0))
    return sum(loglh), loglh
end