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
                   s_non::AbstractMatrix{Float64}, s_init::AbstractMatrix{Float64},
                   ϵ::AbstractMatrix{Float64}, c::Float64, N_MH::Int;
                   ϵ_testing::Matrix{Float64} = zeros(0,0), parallel::Bool = false)
    #------------------------------------------------------------------------
    # Setup
    #------------------------------------------------------------------------

    # Check if testing
    testing = !isempty(ϵ_testing)

    # Sizes
    n_obs    = size(y_t, 1)
    n_states = size(s_init, 1)
    n_shocks = size(ϵ, 1)
    n_particles = size(ϵ, 2)

    # Initialize vector of acceptances
    MyVector = parallel ? SharedVector : Vector
    accept_vec = MyVector{Int}(n_particles)

    #------------------------------------------------------------------------
    # Metropolis-Hastings Steps
    #------------------------------------------------------------------------

    dist = MvNormal(zeros(n_shocks), c^2*QQ)

    @mypar parallel for i in 1:n_particles
        # Generate new ϵ centered at ϵ_init
        ϵ_new = ϵ[:, i] + rand(dist)

        s_non[:,i], ϵ[:,i], accept_vec[i] =
            mh_step(Φ, Ψ, y_t, s_init[:,i], s_non[:,i], ϵ[:,i], ϵ_new,
                    φ_new, det_HH, inv_HH, n_obs, n_shocks, N_MH; testing = testing)
    end

    # Calculate and return acceptance rate
    accept_rate = sum(accept_vec)/(N_MH*n_particles)
    return accept_rate
end

function mh_step(Φ::Function, Ψ::Function, y_t::Vector{Float64}, s_init::Vector{Float64},
                 s_non::Vector{Float64}, ϵ_init::Vector{Float64}, ϵ_new::Vector{Float64},
                 φ_new::Float64, det_HH::Float64, inv_HH::Matrix{Float64},
                 n_obs::Int, n_shocks::Int, N_MH::Int; testing::Bool = false)
    s_out = similar(s_init)
    ϵ_out = similar(ϵ_init)
    accept = 0

    for j = 1:N_MH

        # Use the state equation to calculate the corresponding state from that ε
        s_new = Φ(s_init, ϵ_new)

        # Calculate difference between data and expected y from measurement equation
        error_new  = y_t - Ψ(s_new)
        error_init = y_t - Ψ(s_non)

        # Calculate posteriors
        μ_1 = zeros(n_obs)
        scaled_det_HH = det_HH/(φ_new)^n_obs
        scaled_inv_HH = inv_HH*φ_new
        post_new_1  = fast_mvnormal_pdf(error_new,  μ_1, scaled_det_HH, scaled_inv_HH)
        post_init_1 = fast_mvnormal_pdf(error_init, μ_1, scaled_det_HH, scaled_inv_HH)

        μ_2 = zeros(n_shocks)
        inv_ϵ_cov = eye(n_shocks)
        post_new_2  = fast_mvnormal_pdf(ϵ_new,  μ_2, 1., inv_ϵ_cov)
        post_init_2 = fast_mvnormal_pdf(ϵ_init, μ_2, 1., inv_ϵ_cov)

        post_new  = post_new_1  * post_new_2
        post_init = post_init_1 * post_init_2

        # Calculate α, probability of accepting the new particle
        α = post_new/post_init
        rval = testing ? 0.5 : rand()

        # Accept the particle with probability α
        if rval < α
            # Accept and update particle
            s_out = s_new
            ϵ_out = ϵ_new
            accept += 1
        else
            # Reject and keep particle unchanged
            s_out = s_non
            ϵ_out = ϵ_init
        end
        ϵ_init = ϵ_out
        s_non  = s_out
    end
    return s_out, ϵ_out, accept
end
