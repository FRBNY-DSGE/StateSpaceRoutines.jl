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
function mutation{S<:AbstractFloat}(Φ::Function, Ψ::Function, F_ϵ::Distribution,
                                    F_u::Distribution, φ_new::S, y_t::Vector{S}, s_non::Matrix{S},
                                    s_init::Matrix{S}, ϵ_init::Matrix{S}, c::S, N_MH::Int;
                                    ϵ_testing::Matrix{S} = zeros(0,0), parallel::Bool = false)
    #------------------------------------------------------------------------
    # Setup
    #------------------------------------------------------------------------

    # Check if testing
    testing = !isempty(ϵ_testing)

    # Initialize s_out and ε_out
    s_out = similar(s_init)
    ϵ_out = similar(ϵ_init)

    HH = F_u.Σ.mat

    # Store length of y_t, ε
    n_obs    = size(y_t, 1)
    n_states = size(ϵ_init, 1)
    n_particles = size(ϵ_init, 2)

    # Initialize acceptance counter to zero
    accept_vec = zeros(n_particles)

    #------------------------------------------------------------------------
    # Metropolis-Hastings Steps
    #------------------------------------------------------------------------
    # Generate new draw of ε from a N(ε_init, c²I) distribution, c tuning parameter, I identity
    ϵ_new = !testing ? ϵ_init + c^2 * randn(n_states, n_particles) : ϵ_testing

    if parallel
        out = @sync @parallel (hcat) for i = 1:n_particles
            mh_step(Φ, Ψ, y_t, s_init[:,i], s_non[:,i], ϵ_init[:,i], ϵ_new[:,i], φ_new, HH, n_obs, n_states, N_MH;
                    testing = testing)
        end
        for i = 1:n_particles
            s_out[:,i]    = out[i][1]
            ϵ_out[:,i]    = out[i][2]
            accept_vec[i] = out[i][3]
        end
    else
        for i = 1:n_particles
            s_out[:,i], ϵ_out[:,i], accept_vec[i] = mh_step(Φ, Ψ, y_t, s_init[:,i], s_non[:,i], ϵ_init[:,i],
                                                            ϵ_new[:,i], φ_new, HH, n_obs, n_states, N_MH;
                                                            testing = testing)
        end
    end

    # Calculate acceptance rate
    accept_rate = sum(accept_vec)/(N_MH*n_particles)

    return s_out, ϵ_out, accept_rate
end

function mh_step(Φ::Function, Ψ::Function, y_t::Vector{Float64}, s_init::Vector{Float64},
                 s_non::Vector{Float64}, ϵ_init::Vector{Float64}, ϵ_new::Vector{Float64},
                 φ_new::Float64, HH::Matrix{Float64}, n_obs::Int, n_states::Int, N_MH::Int;
                 testing::Bool = false)
    s_out = similar(s_init)
    ϵ_out = similar(ϵ_init)
    accept = 0.

    for j = 1:N_MH

        # Use the state equation to calculate the corresponding state from that ε
        s_new = Φ(s_init, ϵ_new)

        # Calculate difference between data and expected y from measurement equation
        u_t = zeros(n_obs)
        error_new  = y_t - Ψ(s_new, u_t)
        error_init = y_t - Ψ(s_non, u_t)

        # Calculate posteriors
        A = MvNormal(zeros(n_obs), HH/φ_new)
        B = MvNormal(zeros(n_states), eye(n_states))
        post_new  = pdf(A, error_new) * pdf(B, ϵ_new)
        post_init = pdf(A, error_init) * pdf(B, ϵ_init)

        # Calculate α, probability of accepting the new particle
        α = post_new/post_init
        rval = testing ? 0.5 : rand()

        # Accept the particle with probability α
        if rval < α
            # Accept and update particle
            s_out = s_new
            ϵ_out = ϵ_new
            accept += 1.
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
