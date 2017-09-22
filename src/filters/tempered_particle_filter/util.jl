"""
```
initialize_state_draws(s0::Vector{Float64}, F_ϵ::Distribution, Φ::Function, n_parts::Int;
                       burn::Int = 10000, thin::Int = 5)
```

### Inputs

- `s0::Vector`: An initial guess/starting point to begin iterating the states forward.
- `F_ϵ::Distribution`: The shock distribution: ϵ ~ F_ϵ
- `Φ::Function`: The state transition function: s_t = Φ(s_t-1, ϵ_t)
- `n_parts::Int`: The number of particles (draws) to generate

### Keyword Arguments

- `initialize::Bool`: Flag indicating whether one is solving for incremental weights during
    the initialization of weights; default is `false`.
- `burn::Int`: The number of draws to burn in before the draws are actually collected.
This is under the assumption that the s_t reaches its stationary distribution post burn-in.
- `thin::Int`: The number of draws to thin by to minimize serial correlation

### Output

- `s_init`: A matrix (# of states x # of particles) containing the initial draws of states to start
the tpf algorithm from.
"""
function initialize_state_draws(s0::Vector{Float64}, F_ϵ::Distribution, Φ::Function,
                                n_parts::Int; burn::Int = 10000, thin::Int = 5)
    s_init = zeros(length(s0), n_parts)
    s_old = s0
    for i in 1:(burn + thin*n_parts)
        ϵ = rand(F_ϵ)
        s_new = Φ(s_old, ϵ)

        if i > burn && i % thin == 0
            draw_index = convert(Int, (i - burn)/thin)
            s_init[:, draw_index] = s_new
        end

        s_old = s_new
    end
    return s_init
end

"""
```
update_c!(c_in::Float64, accept_in::Float64, target_in::Float64)
```
Updates value of c by expression that is function of the target and mean acceptance rates.
Returns the new c, in addition to storing it in the model settings.

"""
@inline function update_c!(c_in::Float64, accept_in::Float64, target_in::Float64)
    c_out = c_in*(0.95 + 0.1*exp(20*(accept_in - target_in))/(1 + exp(20*(accept_in - target_in))))
    return c_out
end

"""
```
solve_inefficiency{S<:AbstractFloat}(φ_new::S, φ_old::S, y_t::Vector{S}, p_error::Matrix{S},
inv_HH::Matrix{S}, det_HH::S; initialize::Bool=false)
```
Returns the value of the ineffeciency function InEff(φₙ), where:

        InEff(φₙ) = (1/M) ∑ᴹ (W̃ₜʲ(φₙ))²

Where ∑ is over j=1...M particles, and for a particle j:

        W̃ₜʲ(φₙ) = w̃ₜʲ(φₙ) / (1/M) ∑ᴹ w̃ₜʲ(φₙ)

Where ∑ is over j=1...M particles, and incremental weight is:

        w̃ₜʲ(φₙ) = pₙ(yₜ|sₜʲ'ⁿ⁻¹) / pₙ₋₁(yₜ|sₜ^{j,n-1})
                = (φₙ/φₙ₋₁)^(d/2) exp{-1/2 [yₜ-Ψ(sₜʲ'ⁿ⁻¹)]' (φₙ-φₙ₋₁) ∑ᵤ⁻¹ [yₜ-Ψ(sₜʲ'ⁿ⁻¹)]}

### Inputs

- `φ_new`: φₙ
- `φ_old`: φₙ₋₁
- `y_t`: vector of observables for time t
- `p_error`: (`n_states` x `n_particles`) matrix of particles' errors yₜ - Ψ(sₜʲ'ⁿ⁻¹) in columns
- `inv_HH`: The inverse of the measurement error covariance matrix, ∑ᵤ
- `det_HH`: The determinant of the measurement error covariance matrix, ∑ᵤ

### Keyword Arguments

- `initialize::Bool`: flag to indicate whether this is being used in initialization stage,
    in which case one instead solves the formula for w̃ₜʲ(φₙ) as:

    w̃ₜʲ(φ₁) = (φ₁/2π)^(d/2)|∑ᵤ|^(1/2) exp{-1/2 [yₜ-Ψ(sₜʲ'ⁿ⁻¹)]' φ₁ ∑ᵤ⁻¹ [yₜ-Ψ(sₜʲ'ⁿ⁻¹)]}

"""
function solve_inefficiency{S<:AbstractFloat}(φ_new::S, φ_old::S, y_t::Vector{S},
                                              p_error::Matrix{S}, inv_HH::Matrix{S},
                                              det_HH::S; initialize::Bool = false,
                                              parallel::Bool = false)

    n_particles = size(p_error, 2)
    n_obs       = length(y_t)

    # Inefficiency function during initialization
    if initialize
        c = (φ_new/(2*pi))^(n_obs/2) * (det_HH^(-1/2))
        if parallel
            w = @parallel (vcat) for i = 1:n_particles
                c*exp(-1/2 * φ_new * dot(p_error[:,i], inv_HH * p_error[:,i]))
            end
        else
            w = zeros(n_particles)
            for i = 1:n_particles
                w[i] = c*exp(-1/2 * φ_new * dot(p_error[:,i], inv_HH * p_error[:,i]))
            end
        end

    # Inefficiency function during tempering steps
    else
        c = (φ_new/φ_old)^(n_obs/2)
        if parallel
            w = @parallel (vcat) for i = 1:n_particles
                c*exp(-1/2 * (φ_new-φ_old) * dot(p_error[:,i], inv_HH * p_error[:,i]))
            end
        else
            w = zeros(n_particles)
            for i = 1:n_particles
                w[i] = c*exp(-1/2 * (φ_new-φ_old) * dot(p_error[:,i], inv_HH * p_error[:,i]))
            end
        end
    end
    W = w/mean(w)
    return sum(W.^2)/n_particles
end

"""
```
incremental_weight(φ_new::Float64, φ_old::Float64, y_t::Vector{Float64}, p_error::Vector{Float64},
HH::Matrix{Float64}; initialize::Bool=false)
```

### Inputs

- `φ_new::Float64`: current φ
- `φ_old::Float64`: φ value before last
- `y_t::Vector{Float64}`: Vector of observables for time t
- `p_error::Vector{Float64}`: A single particle's error: y_t - Ψ(s_t)
- `HH::Matrix{Float64}`: Measurement error covariance matrix

### Keyword Arguments
- `initialize::Bool`: Flag indicating whether one is solving for incremental weights during
    the initialization of weights; default is `false`.

### Output
- Returns the incremental weight of single particle
"""
@inline function incremental_weight(φ_new::Float64, φ_old::Float64, y_t::Vector{Float64},
                                    p_error::Vector{Float64}, HH::Matrix{Float64}; initialize::Bool = false)

    # Initialization step (using 2π instead of φ_old)
    if initialize
        return (φ_new/(2*pi))^(length(y_t)/2) * (det(HH)^(-1/2)) *
            exp(-1/2 * p_error' * φ_new * inv(HH) * p_error)[1]

    # Non-initialization step (tempering and final iteration)
    else
        return (φ_new/φ_old)^(length(y_t)/2) *
            exp(-1/2 * p_error' * (φ_new - φ_old) * inv(HH) * p_error)[1]
    end
end

"""
```
resample(weights::AbstractArray; method::Symbol=:systematic,
         parallel::Bool=false, testing::Bool=false)
```

Reindexing and reweighting samples from a degenerate distribution

### Arguments:

- `weights`: The weights of each of the particles.
- `method`: :systematic or :multinomial
        the method for resampling.
- `parallel`: to indicate whether to resample using multiple workers (if available).
- `testing`: to indicate whether to give test output.

### Output:

- `vec(indx)`: id
        the newly assigned indices of parameter draws.
"""
function resample(weights::AbstractArray; method::Symbol = :multinomial,
                  parallel::Bool = false, testing::Bool = false)
    if method == :systematic
        n_parts = length(weights)
        # Stores cumulative weights until given index
        cumulative_weights = cumsum(weights./sum(weights))
        uu = Vector{Float64}(n_parts)

        # Random part of algorithm - choose offset of first index by some u~U[0,1)
        rand_offset = rand()

        # Set "spokes" at the position of the random offset
        for j = 1:n_parts
            uu[j] = (j - 1) + rand_offset
        end

        # Function solves where an individual "spoke" lands
        function subsys(i::Int)
            findfirst(j -> cumulative_weights[j] > uu[i]/n_parts, 1:length(cumulative_weights))
        end

        # Map function if parallel
        if parallel
            indx =
            @sync @parallel (vcat) for j in 1:n_parts
                subsys(j)
            end
        else
            indx = [subsys(j) for j = 1:n_parts]
        end

        return indx

    elseif method == :multinomial
        n_parts = length(weights)
        weights = Weights(weights./sum(weights))

        return sample(1:n_parts, weights, n_parts, replace = true)
    else
        throw("Invalid resampler. Set tuning field :resampling_method to either :systematic or :multinomial")
    end
end
