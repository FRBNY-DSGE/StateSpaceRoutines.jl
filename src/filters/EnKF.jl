"""
```
tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; verbose = :high,
    n_particles = 1000, fixed_sched = [], r_star = 2, findroot = bisection,
    xtol = 1e-3, resampling_method = :multionial, n_mh_steps = 1, c_init = 0.3,
    target_accept_rate = 0.4, n_presample_periods = 0, allout = true,
    parallel = false, verbose = :low,
    dynamic_measurement = false, poolmodel = false)
```

### Inputs

- `data::Matrix{S}`: `Ny` x `Nt` matrix of historical data
- `Φ::Function`: state transition equation: s_t = Φ(s_{t-1}, ϵ_t)
- `Ψ::Function`: measurement equation: y_t = Ψ(s_t) + u_t
- `F_ϵ::Distribution`: shock distribution: ϵ_t ~ F_ϵ
- `F_u::Distribution`: measurement error distribution: u_t ~ F_u
- `s_init::Matrix{S}`: `Ns` x `n_particles` matrix of initial state vectors s_0

where `S<:AbstractFloat` and

- `Ns` is the number of states
- `Ny` is the number of observables
- `Nt` is the number of data periods

### Keyword Arguments

- `n_particles::Int`: number of particles used to approximate the
  log-likelihood. Using more particles yields a more accurate estimate, at the
  cost of being more computationally intensive

**Correction:**

- `fixed_sched::Vector{S}`: array of monotone increasing values in (0,1] which
  specify the fixed tempering schedule. Set to `[]` if you wish to adaptively
  choose the tempering schedule (below)
- `r_star::S`: target inefficiency ratio used to adaptively choose φ, i.e.
  Ineff(φ_n) := 1/M Σ_{j=1}^M (Wtil_t^{j,n}(φ_n))^2 = r_star
- `findroot::Function`: root-finding algorithm, one of `bisection` or `fzero`
- `xtol::S`: root-finding tolerance on x bracket size

**Selection:**

- `resampling_method::Symbol`: either `:multinomial` or `:systematic`

**Mutation:**

- `n_mh_steps::Int`: number of Metropolis-Hastings steps to take in each
  tempering stage
- `c_init::S`: initial scaling of proposal covariance matrix
- `target_accept_rate::S`: target Metropolis-Hastings acceptance rate

**Other:**

- `n_presample_periods::Int`: number of initial periods to omit from the
  log-likelihood calculation
- `allout::Bool`: whether to return all outputs or just `sum(loglh)`
- `parallel::Bool`: whether to use `SharedArray`s
- `verbose::Symbol`: amount to print to STDOUT. One of `:none`, `:low`, or
  `:high`

### Outputs

- `sum(loglh)::S`: log-likelihood p(y_{1:T}|Φ,Ψ,F_ϵ,F_u) approximation
- `loglh::Vector{S}`: vector of conditional log-likelihoods p(y_t|y_{1:t-1},Φ,Ψ,F_ϵ,F_u)
- `times::Vector{S}`: vector of runtimes per period t
"""
## TODO: Allow Φ and Ψ to vary over time (do the same in TPF), just like in Kalman Filter.
function ensemble_kalman_filter(data::AbstractArray, Φ::Function, Ψ::Function,
                                  F_ϵ::Distribution, F_u::Distribution,
                                  s_init::AbstractArray{S}; n_particles::Int = 100,
                                  n_presample_periods::Int = 0, allout::Bool = true,
                                  parallel::Bool = false, get_t_particle_dist::Bool = false,
                                  verbose::Symbol = :high) where S<:AbstractFloat

    #--------------------------------------------------------------
    # Setup
    #--------------------------------------------------------------

    # Initialize output vectors
    loglh = zeros(T)
    times = zeros(T)

    # Initialize matrix of normalized weight per particle by time period
    # and a matrix of the particle locations, if desired by user
    if get_t_particle_dist
        t_particle_dist = Dict{Int64,Matrix{Float64}}()
    end

    # Initialize working variables
    n_states  = size(s_init, 1)

    if n_states == 1
        s_t_nontemp   = parallel ? dzeros(n_particles) :
            Vector{Float64}(undef, n_particles)
        ϵ_t           = parallel ? dzeros(n_particles) :
            Vector{Float64}(undef, n_particles)
    else
        s_t_nontemp   = parallel ? dzeros((n_states, n_particles), workers(), [1,nworkers()]) :
            Matrix{Float64}(undef, n_states, n_particles)
        ϵ_t           = parallel ? dzeros((n_shocks, n_particles), workers(), [1,nworkers()]) :
            Matrix{Float64}(undef, n_shocks, n_particles)
    end

    #--------------------------------------------------------------
    # Main Algorithm: Tempered Particle Filter
    #--------------------------------------------------------------

    for t = 1:T
        begin_time = time_ns()
        if VERBOSITY[verbose] >= VERBOSITY[:low]
            println("============================================================")
            @show t
        end

        #--------------------------------------------------------------
        # Initialization
        #--------------------------------------------------------------

        y_t = data[:, t]

        # Adjust other values to remove rows/columns with NaN values
        nonmissing = isfinite.(y_t)
        y_t        = y_t[nonmissing]
        n_obs_t    = length(y_t)

        # Remove rows/columns of series with NaN values
        # Handle measurement equation
        Ψ_t  = x -> Ψ(x)[nonmissing]
        if parallel
            sendto(workers(), Ψ_t = Ψ_t)
        end

        if parallel
            ϵ_t_vec = rand(F_ϵ, n_particles)

            if n_states == 1
                ϵ_t = distribute(ϵ_t_vec)

                @sync @distributed for w in workers()
                    s_t_nontemp[:L][:] .= Φ.(s_t_nontemp[:L], ϵ_t[:L])
                end
            else
                if ndims(ϵ_t_vec) == 1 # Edge case where only 1 shock
                    ϵ_t_vec = reshape(ϵ_t_vec, (1, length(ϵ_t_vec)))
                end
                ϵ_t = distribute(ϵ_t_vec, dist = [1, nworkers()])

                @sync @distributed for w in workers()
                    for i in 1:size(s_t_nontemp[:L],2)
                        s_t_nontemp[:L][:,i] = Φ(s_t_nontemp[:L][:,i], ϵ_t[:L][:,i])
                    end
                end
            end
        else
            # Step 0: For later use
            avg_wts = fill(1.0 / n_particles, (n_particles, n_particles))
            update_prod = I - avg_wts

            # Step 1: Predict
            ##TODO: Need pseudoinverse for regular EnKF (not TEnKF)
            ##TODO: Do transposes if using TEnKF
             for i in 1:n_particles
                 ϵ_t[:, i] .= rand(F_ϵ)
                 s_t_nontemp[:, i] = Φ(s_t1_temp[:, i], ϵ_t[:, i])

                 Z_t_t1 = Ψ_t(s_t_nontemp[:,i]) + rand(F_u) ##TODO: Use Ψ_allstates as appropriate
                 ## This is different from econsieve and more in line with the equation
                 ## econsieve doesn't add measurement error to Z-bar.
                 ## In expectation, there is no difference.
            end

            # Step 2: Update
            Xbar = s_t_nontemp * update_prod
            Zbar = Z_t_t1 * update_prod
            Zcov = Zbar * Zbar'

            s_t_nontemp .+= Xbar * Zbar' * inv(Zcov) * (y_t .- Z_t_t1)

            # Step 3: Log Likelihood Update
            diff = y_t .- mean(Z_t_t1, dims = 1)
            loglh[t] = logpdf(MvNormal(Zcov), diff)
        end

        if get_t_particle_dist
            t_particle_dist[t] = copy(s_t_nontemp)
        end

        times[t] = time_ns() - begin_time

        if VERBOSITY[verbose] >= VERBOSITY[:low]
            print("\n")
            @show loglh[t]
            print("Completion of one period $times[t]")
        end
    end

    if VERBOSITY[verbose] >= VERBOSITY[:low]
        println("=============================================")
    end

    if get_t_particle_dist && allout
        return sum(loglh[n_presample_periods + 1:end]), loglh[n_presample_periods + 1:end], times, t_particle_dist
    elseif get_t_particle_dist
        return sum(loglh[n_presample_periods + 1:end]), t_particle_dist
    elseif allout
        return sum(loglh[n_presample_periods + 1:end]), loglh[n_presample_periods + 1:end], times
    else
        return sum(loglh[n_presample_periods + 1:end])
    end
end
