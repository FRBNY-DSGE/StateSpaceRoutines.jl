# s_t = TT*s_{t-1} + RR*ε_t, ε_t ~ iidN(0, QQ)
# y_t = DD+ ZZ*s_t + η_t, η_t ~ iidN(0, HH)
#y is nobs x ny with each row being time period

function chand_recursion(y::Matrix{S},
                         T::Matrix{S}, R::Matrix{S}, C::Vector{S},
                         Q::Matrix{S}, Z::Matrix{S}, D::Vector{S}, H::Matrix{S},
                         s_pred::Vector{S} = Vector{S}(undef, 0), P_pred::Matrix{S} = Matrix{S}(undef, 0,0);
                         allout::Bool = true, Nt0::Int = 0) where {S<:AbstractFloat}
    # Dimensions
    Ns = size(T, 1) # number of states
    Ny, Nt = size(y) # number of periods of data

    # Initialize s_pred and P_pred
    if isempty(s_pred) || isempty(P_pred)
        s_pred, P_pred = init_stationary_states(T, R, C, Q)
    end

    # V_{t|t-1} = Var(y_t|y_{1:t-1})
    V_pred = Z*P_pred*Z' + H
    V_pred = 0.5*(V_pred+V_pred')
    invV_pred = inv(V_pred)

    invV_t1 = invV_pred

    # We write ΔP in terms of W_t and M_t           ΔP = W_t * M_t * W_t'
    W_t = T*P_pred*Z' # (Eq 12)
    M_t = -invV_pred    # (Eq 13)
    kal_gain = W_t*invV_pred

    # Initialize loglikelihoods to zeros so that the ones we don't update (those in presample) contribute zero towards sum of loglikelihoods.
    loglh = Vector{Float64}(undef, Nt)
    zero_vec = zeros(Ny)
    ν_t = zero_vec
    P =  P_pred

    for t in 1:Nt
        # Step 1: Compute forecast error, ν_t and evaluate likelihoood
        yhat = Z*s_pred + D
        ν_t = y[:, t] - yhat
        loglh[t] = logpdf(MvNormal(zero_vec, V_pred), ν_t)

        # Step 2: Compute s_{t+1} using Eq. 5
        if t < Nt
            s_pred = T*s_pred + kal_gain*ν_t
            if allout
                P = P + W_t*M_t*W_t'
            end
        end

        # Intermediate calculations to re-use
        ZW_t = Z*W_t
        MWpZp = M_t*(ZW_t')
        TW_t = T*W_t

        # Step 3: Update forecast error variance F_{t+1} (Eq 19)
        V_t1 = V_pred
        invV_t1 = invV_pred
        V_pred  = V_pred + ZW_t*MWpZp        # F_{t+1}
        V_pred  = 0.5*(V_pred+V_pred')
        invV_pred = inv(V_pred)

        # Step 4: Update Kalman Gain (Eq 20). Recall that kalgain = K_t * V_t⁻¹
        # Kalgain_{t+1} = (Kalgain_t*V_{t-1} + T*W_t*M_t*W_t'*Z')V_t⁻¹
        kal_gain = (kal_gain*V_t1 + TW_t*MWpZp)*invV_pred

        # Step 5: Update W
        W_t = TW_t - kal_gain*ZW_t    # W_{t+1}

        # Step 6: Update M
        M_t = M_t + MWpZp*invV_t1*MWpZp'     # M_{t+1}
        M_t = 0.5*(M_t + M_t')
    end
    loglh = remove_presample!(Nt0, loglh)
    if allout
        s_TT = s_pred + P'*Z'*invV_t1*(ν_t)
        P_TT = P - P'*Z'*invV_t1*Z*P
        return sum(loglh), loglh, s_TT, P_TT
    else
        return loglh
    end
end

function remove_presample!(Nt0::Int, loglh::Vector{S}) where {S<:AbstractFloat}
    out = remove_presample!(Nt0, loglh, Array{Float64, 2}(undef, 0,0), Array{Float64,3}(undef, 0,0,0), Array{Float64, 2}(undef, 0,0), Array{Float64, 3}(undef, 0,0,0), outputs = [:loglh])
    return out[1]
end

#=function chand_recursion(regime_indices::Vector{Range{Int}}, y::Matrix{S},
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
end =#
