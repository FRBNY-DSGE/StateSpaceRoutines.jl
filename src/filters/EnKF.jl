"""
```
ensemble_kalman_filter(data, Φ, Ψ, F_ϵ, F_u, s_init;
    n_particles = 1000, n_presample_periods = 0,
    allout = true, get_t_particle_dist::Bool = false,
    parallel = false, verbose = :none)
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
                                get_t_particle_dist::Bool = false,
                                parallel::Bool = false, verbose::Symbol = :none) where S<:AbstractFloat

    #--------------------------------------------------------------
    # Setup
    #--------------------------------------------------------------

    # Initialize working variables
    n_states  = length(size(s_init)) == 1 ? 1 : size(s_init, 1)
    n_obs     = length(F_u)
    n_shocks = length(F_ϵ)

    # Initialize output vectors
    T = n_obs == 1 ? length(data) : size(data,2)
    loglh = zeros(T)
    times = zeros(T)

    # Initialize matrix of normalized weight per particle by time period
    # and a matrix of the particle locations, if desired by user
    if get_t_particle_dist
        t_particle_dist = n_states == 1 ? Dict{Int64,Vector{Float64}}() : Dict{Int64,Matrix{Float64}}()
    end

    if n_states == 1
        s_t_nontemp   = copy(vec(s_init))
        ϵ_t           = Vector{Float64}(undef, n_particles)
    else
        s_t_nontemp   = copy(s_init)
        ϵ_t           = Matrix{Float64}(undef, n_shocks, n_particles)
    end

    if n_obs == 1
        Z_t_t1        = Vector{Float64}(undef, n_particles)
    else
        Z_t_t1        = Matrix{Float64}(undef, n_obs, n_particles)
    end

    Xbar = similar(s_t_nontemp)
    Zbar = similar(Z_t_t1)
    Zcov = n_obs == 1 ? 1.0 : Matrix{Float64}(undef, n_obs, n_obs)

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

        if (length(size(data)) == 1) || (size(data,1) == 1)
            y_t = data[t] ## Assume not missing
            Ψ_t  = x -> Ψ(x)
        else
            y_t = data[:, t]

            # Adjust other values to remove rows/columns with NaN values
            nonmissing = isfinite.(y_t)
            y_t        = y_t[nonmissing]
            n_obs_t    = length(y_t)

            # Remove rows/columns of series with NaN values
            # Handle measurement equation
            Ψ_t  = x -> Ψ(x)[nonmissing]
        end

        # Step 0: For later use
        if n_states > 1 || n_obs > 1
            avg_wts = fill(1.0 / n_particles, (n_particles, n_particles))
            update_prod = I - avg_wts
        end

        # Step 1: Predict
        ϵ_t = rand(F_ϵ, n_particles)
        if n_states == 1
            s_t_nontemp = Φ.(s_t_nontemp, ϵ_t)
            if n_obs == 1
                Z_t_t1 = Ψ_t.(s_t_nontemp)
            else
                for i in 1:n_particles
                    Z_t_t1[:,i] = Ψ_t(s_t_nontemp[i])
                end
             end
        elseif n_obs == 1
            for i in 1:n_particles
                s_t_nontemp[:, i] = Φ(s_t_nontemp[:, i], ϵ_t[:, i])
                Z_t_t1[i] = Ψ_t(s_t_nontemp[:,i])
            end
        else
            for i in 1:n_particles
                s_t_nontemp[:, i] = Φ(s_t_nontemp[:, i], ϵ_t[:, i])
                Z_t_t1[:,i] = Ψ_t(s_t_nontemp[:,i])
            end
        end

        # Step 2: Update
        if n_states == 1
            Xbar = s_t_nontemp .- mean(s_t_nontemp)
        else
            mul!(Xbar, s_t_nontemp, update_prod)
        end

        if n_obs == 1
            Zbar = Z_t_t1 .- mean(Z_t_t1)
            Zcov = var(Zbar) + var(F_u)
        else
            mul!(Zbar, Z_t_t1, update_prod)
            Zcov = cov(Z_t_t1, dims = 2) .+ cov(F_u)
        end

        # s_t_nontemp .+= Xbar * Zbar' * (Zcov \ (y_t .- Z_t_t1))
        if n_states == 1 && n_obs == 1
            s_t_nontemp .+= Xbar .* Zbar .* ((y_t .- Z_t_t1 .- rand(F_u,n_particles)) ./ ((n_particles - 1) * Zcov))
        elseif n_states == 1
            obs_mat = Zbar' * (((n_particles - 1) * Zcov) \ (y_t .- Z_t_t1 .- rand(F_u,n_particles)))
            # s_t_nontemp .+= vec(Xbar' * Zbar' * (((n_particles - 1) * Zcov) \ (y_t .- Z_t_t1 .- rand(F_u,n_particles))))
                @inbounds @simd for i = 1:length(s_t_nontemp)
                    s_t_nontemp[i] += dot(Xbar, obs_mat[:,i])
                end
        elseif n_obs == 1
            s_t_nontemp .+= Xbar * (Zbar .* ((y_t .- Z_t_t1 .- rand(F_u,n_particles)) ./ ((n_particles - 1) * Zcov)))
        else
            s_t_nontemp .+= Xbar * Zbar' * (((n_particles - 1) * Zcov) \ (y_t .- Z_t_t1 .- rand(F_u,n_particles)))
        end
        ## Backslash operator checks singularity (at least checks if matrix is rectangular) so TEnKF = EnKF

        # Step 3: Log Likelihood Update
        diff = n_obs == 1 ? y_t - mean(Z_t_t1) : y_t .- vec(mean(Z_t_t1, dims = 2))
        loglh[t] = n_obs == 1 ? logpdf(Normal(Zcov), diff) : logpdf(MvNormal(Zcov),diff)

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
