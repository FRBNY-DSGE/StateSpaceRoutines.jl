"""
```
remove_presample!(Nt0, loglh, s_pred, P_pred, s_filt, P_filt)
```

Remove the first `Nt0` periods from all other input arguments and return.
"""
function remove_presample!(Nt0::Int, loglh::Vector{S},
                           s_pred::Matrix{S}, P_pred::Array{S, 3},
                           s_filt::Matrix{S}, P_filt::Array{S, 3};
                           outputs::Vector{Symbol} = [:loglh, :pred, :filt]) where {S<:AbstractFloat}
    if Nt0 > 0
        if :loglh in outputs
            loglh  = loglh[(Nt0+1):end]
        end
        if :pred in outputs
            s_pred = s_pred[:,    (Nt0+1):end]
            P_pred = P_pred[:, :, (Nt0+1):end]
        end
        if :filt in outputs
            s_filt = s_filt[:,    (Nt0+1):end]
            P_filt = P_filt[:, :, (Nt0+1):end]
        end
    end
    return loglh, s_pred, P_pred, s_filt, P_filt
end

function remove_presample!(Nt0::Int, loglh::Vector{S}) where {S<:AbstractFloat}
    if Nt0 > 0
        return loglh[(Nt0+1):end]
    end
    return loglh
end
