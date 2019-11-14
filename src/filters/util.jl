"""
```
remove_presample!(Nt0, loglh, s_pred, P_pred, s_filt, P_filt)
```

Remove the first `Nt0` periods from all other input arguments and return.
"""
function remove_presample!(Nt0::Int, loglh::Vector{S},
                           s_pred::Matrix{S}, P_pred::Array{S, 3},
                           s_filt::Matrix{S}, P_filt::Array{S, 3};
                           outputs::Vector{Symbol} = [:loglh, :pred, :filt]) where {S<:Real}
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

function remove_presample!(Nt0::Int, loglh::Vector{S}) where {S<:Real}
    if Nt0 > 0
        return loglh[(Nt0+1):end]
    end
    return loglh
end

"""
```
solve_discrete_lyapunov(A, B, max_it = 50)
```
Solves the discrete Lyapunov equation.

The problem is given by

```
    AXA' - X + B = 0
```

``X`` is computed by using a doubling algorithm. In particular, we iterate to
convergence on ``X_j`` with the following recursions for ``j = 1, 2, ...``
starting from ``X_0 = B, a_0 = A``:

```
    a_j = a_{j-1} a_{j-1}
    X_j = X_{j-1} + a_{j-1} X_{j-1} a_{j-1}'
```

This function is directly copied and pasted from the QuantEcon.jl package
and extended to handle Flux.Tracker through multiple dispatch.

##### Arguments

- `A::Matrix{Float64}` : An `n x n` matrix as described above.  We assume in order
  for  convergence that the eigenvalues of ``A`` have moduli bounded by unity
- `B::Matrix{Float64}` :  An `n x n` matrix as described above.  We assume in order
  for convergence that the eigenvalues of ``B`` have moduli bounded by unity
- `max_it::Int(50)` :  Maximum number of iterations

##### Returns

- `gamma1::Matrix{Float64}` Represents the value ``X``

"""
function solve_discrete_lyapunov(A::Matrix{S},
                                 B::Matrix{S},
                                 max_it::Int=50) where {S<:Real}
    # TODO: Implement Bartels-Stewardt
    n = size(A, 2)
    alpha0 = reshape([A;], n, n)
    gamma0 = reshape([B;], n, n)

    alpha1 = fill!(similar(alpha0), zero(eltype(alpha0)))
    gamma1 = fill!(similar(gamma0), zero(eltype(gamma0)))

    diff = 5
    n_its = 1

    while diff > 1e-15

        alpha1 = alpha0*alpha0
        gamma1 = gamma0 + alpha0*gamma0*alpha0'

        diff = maximum(abs, gamma1 - gamma0)
        alpha0 = alpha1
        gamma0 = gamma1

        n_its += 1

        if n_its > max_it
            error("Exceeded maximum iterations, check input matrices")
        end
    end

    return gamma1
end

function solve_discrete_lyapunov(A::TrackedArray{S},
                                 B::TrackedArray{S},
                                 max_it::Int=50) where {S<:Real}
    # TODO: Implement Bartels-Stewardt
    n = size(A, 2)
    alpha0 = reshape([A;], n, n)
    gamma0 = reshape([B;], n, n)

    alpha1 = fill!(similar(alpha0), zero(return_tracker_parameter_type(alpha0)))
    gamma1 = fill!(similar(gamma0), zero(return_tracker_parameter_type(gamma0)))

    diff = 5
    n_its = 1

    while diff > 1e-15

        alpha1 = alpha0*alpha0
        gamma1 = gamma0 + alpha0*gamma0*alpha0'

        diff = maximum(abs, gamma1 - gamma0)
        alpha0 = alpha1
        gamma0 = gamma1

        n_its += 1

        if n_its > max_it
            error("Exceeded maximum iterations, check input matrices")
        end
    end

    return gamma1
end

"""
```
return_tracker_parameter_type(x)
```
returns the underlying type for a Tracker variable.

```jldoctest
julia> return_tracker_type(param(1.0))
Float64
```
"""
return_tracker_parameter_type(x::Tracker.TrackedReal{S}) where S<:Real = S
return_tracker_parameter_type(x::TrackedArray{S}) where S<:Real = S
