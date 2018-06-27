"""
```
mutation{S<:AbstractFloat}(system::System{S}, y_t::Array{S,1}, s_init::Array{S,1},
        ϵ_init::Array{S,1}, c::S, N_MH::Int, nonmissing::Array{Bool,1})
```
Runs random-walk Metropolis Hastings for single particle. The caller should loop through
all particles, calling this method on each.

### Inputs

- `system`: state-space system matrices
- `y_t`: vector of observables at time t
- `s_init`: vector of starting state before mutation (ŝ in paper)
- `ϵ_init`: vector of starting state error before mutation
- `c`: scaling factor used to achieve a desired acceptance rate, adjusted via:

    cₙ = cₙ₋₁f(1-R̂ₙ₋₁(cₙ₋₁))

    Where c₁ = c_star and R̂ₙ₋₁(cₙ₋₁) is the emprical rejection rate based on mutation
    phase in iteration n-1. Average is computed across previous N_MH RWMH steps.

- `N_MH`: number of Metropolis Hastings steps
- `nonmissing`: vector of booleans used to remove NaN values from matrices in system object

### Outputs

- `s_out`: mutated state vector
- `ϵ_out`: output ϵ shock corresponding to state vector
- `accept_rate`: acceptance rate across N_MH steps

"""
function mutation!(Φ::Function, Ψ::Function, QQ::Matrix{Float64},
                   det_HH::Float64, inv_HH::Matrix{Float64}, φ_new::Float64, y_t::Vector{Float64},
                   s_t::M, s_t1::M, ϵ_t::M, c::Float64, N_MH::Int;
                   parallel::Bool = false) where M<:AbstractMatrix{Float64}
    # Sizes
    n_obs = size(y_t, 1)
    n_particles = size(ϵ_t, 2)

    # Initialize vector of acceptances
    MyVector = parallel ? SharedVector : Vector
    accept_vec = MyVector{Int}(n_particles)

    # Used to generate new draws of ϵ
    dist_ϵ = MvNormal(c^2 * diag(QQ))

    # Used to calculate posteriors
    scaled_det_HH = det_HH/(φ_new^n_obs)
    scaled_inv_HH = inv_HH*φ_new

    # Take Metropolis-Hastings steps
    @mypar parallel for i in 1:n_particles
        s_t[:,i], ϵ_t[:,i], accept_vec[i] =
            mh_steps(Φ, Ψ, dist_ϵ, y_t, s_t1[:,i], s_t[:,i], ϵ_t[:,i],
                    scaled_det_HH, scaled_inv_HH, N_MH)
    end

    # Calculate and return acceptance rate
    accept_rate = sum(accept_vec) / (N_MH*n_particles)
    return accept_rate
end

function mh_steps(Φ::Function, Ψ::Function, dist_ϵ::MvNormal, y_t::Vector{Float64},
                  s_t1::Vector{Float64}, s_t::Vector{Float64}, ϵ_t::Vector{Float64},
                  scaled_det_HH::Float64, scaled_inv_HH::Matrix{Float64}, N_MH::Int)
    accept = 0

    # Compute posterior at initial ϵ_t
    post_1 = fast_mvnormal_pdf(y_t - Ψ(s_t), scaled_det_HH, scaled_inv_HH)
    post_2 = fast_mvnormal_pdf(ϵ_t)
    post = post_1 * post_2

    for j = 1:N_MH
        # Draw ϵ_new and s_new
        ϵ_new = ϵ_t + rand(dist_ϵ)
        s_new = Φ(s_t1, ϵ_new)

        # Calculate posterior
        post_new_1 = fast_mvnormal_pdf(y_t - Ψ(s_new), scaled_det_HH, scaled_inv_HH)
        post_new_2 = fast_mvnormal_pdf(ϵ_new)
        post_new = post_new_1 * post_new_2

        # Calculate α, probability of accepting the new particle
        α = post_new / post
        if rand() < α
            s_t = s_new
            ϵ_t = ϵ_new
            post = post_new
            accept += 1
        end
    end
    return s_t, ϵ_t, accept
end

"""
```
update_c(c, accept_rate, target_rate)
```

Return the new proposal covariance matrix scaling `c`, adaptively chosen given
`accept_rate` to match `target_rate`.
"""
@inline function update_c(c::Float64, accept_rate::Float64, target_rate::Float64)
    c*(0.95 + 0.1*exp(20*(accept_rate - target_rate))/(1 + exp(20*(accept_rate - target_rate))))
end