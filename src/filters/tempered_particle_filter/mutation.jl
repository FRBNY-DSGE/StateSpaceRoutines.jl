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
                   s_t::AbstractMatrix{Float64}, s_t1::AbstractMatrix{Float64},
                   ϵ_t::AbstractMatrix{Float64}, c::Float64, N_MH::Int;
                   ϵ_testing::Matrix{Float64} = zeros(0,0), parallel::Bool = false)
    # Check if testing
    testing = !isempty(ϵ_testing)

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
            mh_step(Φ, Ψ, dist_ϵ, y_t, s_t1[:,i], s_t[:,i], ϵ_t[:,i],
                    scaled_det_HH, scaled_inv_HH, N_MH; testing = testing)
    end

    # Calculate and return acceptance rate
    accept_rate = sum(accept_vec) / (N_MH*n_particles)
    return accept_rate
end

function mh_step(Φ::Function, Ψ::Function, dist_ϵ::MvNormal, y_t::Vector{Float64},
                 s_t1::Vector{Float64}, s_t::Vector{Float64}, ϵ_t::Vector{Float64},
                 scaled_det_HH::Float64, scaled_inv_HH::Matrix{Float64},
                 N_MH::Int; testing::Bool = false)
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
        rval = testing ? 0.5 : rand()

        # Accept the particle with probability α
        if rval < α
            s_t = s_new
            ϵ_t = ϵ_new
            post = post_new
            accept += 1
        end
    end
    return s_t, ϵ_t, accept
end
