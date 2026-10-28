using BenchmarkTools, JLD2, HDF5, Test, StateSpaceRoutines
import JLD2: @load
path = dirname(@__FILE__)
run_benchmarks = false

# Initialize arguments to function
@load "$path/reference/kalman_filter_args.jld2" y T R C Q Z D E z0 P0

# Kalman Filter (all arguments and no presample)
out = kalman_filter(y, T, R, C, Q, Z, D, E, z0, P0)
run_benchmarks && @btime kalman_filter($y, $T, $R, $C, $Q, $Z, $D, $E, $z0, $P0)
@testset "Basic Kalman Filter (all arguments, no presample)" begin
    h5open("$path/reference/kalman_filter_out.h5", "r") do h5
        @test read(h5, "log_likelihood") ≈ sum(out[1])
        @test read(h5, "marginal_loglh") ≈ out[1]
        @test read(h5, "pred")           ≈ out[2]
        @test read(h5, "vpred")          ≈ out[3]
        @test read(h5, "filt")           ≈ out[4]
        @test read(h5, "vfilt")          ≈ out[5]
    end
end

# Kalman Filter (no initial conditions and no presample)
out = kalman_filter(y, T, R, C, Q, Z, D, E)
run_benchmarks && @btime kalman_filter($y, $T, $R, $C, $Q, $Z, $D, $E)
@testset "Kalman Filter (no initial conditions)" begin
    h5open("$path/reference/kalman_filter_out.h5", "r") do h5
        @test read(h5, "log_likelihood") ≈ sum(out[1])
        @test read(h5, "marginal_loglh") ≈ out[1]
        @test read(h5, "pred")           ≈ out[2]
        @test read(h5, "vpred")          ≈ out[3]
        @test read(h5, "filt")           ≈ out[4]
        @test read(h5, "vfilt")          ≈ out[5]
        @test z0                         ≈ out[6]
        @test P0                         ≈ out[7]
    end
end

# Kalman filter with presample
out = kalman_filter(y, T, R, C, Q, Z, D, E, Nt0=4)
run_benchmarks && @btime kalman_filter($y, $T, $R, $C, $Q, $Z, $D, $E, Nt0=4)
@testset "Kalman Filter (presample)" begin
    h5open("$path/reference/kalman_filter_out_presample.h5", "r") do h5
        @test read(h5, "log_likelihood") ≈ sum(out[1])
        @test read(h5, "marginal_loglh") ≈ out[1]
        @test read(h5, "pred")           ≈ out[2]
        @test read(h5, "vpred")          ≈ out[3]
        @test read(h5, "filt")           ≈ out[4]
        @test read(h5, "vfilt")          ≈ out[5]
    end
end

# Initialize arguments to for multi-regime Kalman Filter (ZLB)
@load "$path/reference/kalman_filter_args_zlb.jld2" y Ts Rs Cs Qs Zs Ds Es regime_inds

out = kalman_filter(regime_inds, y, Ts, Rs, Cs, Qs, Zs, Ds, Es)
run_benchmarks && @btime kalman_filter($regime_inds, $y, $Ts, $Rs, $Cs, $Qs, $Zs, $Ds, $Es)
@testset "Kalman Filter (Multi-regime/ZLB)" begin
    h5open("$path/reference/kalman_filter_out_zlb.h5", "r") do h5
        @test read(h5, "log_likelihood") ≈ sum(out[1])
        @test read(h5, "marginal_loglh") ≈ out[1]
        @test read(h5, "pred")           ≈ out[2]
        @test read(h5, "vpred")          ≈ out[3]
        @test read(h5, "filt")           ≈ out[4]
        @test read(h5, "vfilt")          ≈ out[5]
    end
end

# Non-PD V_pred guard: a draw whose predicted-observation covariance
# V_pred = Z*P_pred*Z' + E is not positive-definite must yield loglh = -Inf
# (particle rejected), NOT a garbage-but-finite likelihood or a DomainError
# from log(det(V_pred<0)). Regression test for the Vpred guard in update!
# (src/filters/kalman_filter.jl). Feeds update! a P_pred/E that force a non-PD
# V_pred directly, since no reference system in this file exercises that path.
@testset "Kalman Filter (non-PD V_pred guard)" begin
    T   = reshape([0.5], 1, 1); R = reshape([1.0], 1, 1); C = [0.0]
    Q   = reshape([1.0], 1, 1); Z = reshape([1.0], 1, 1); D = [0.0]
    s_0 = [0.0];                 P_0 = reshape([1.0], 1, 1)

    # E has a negative eigenvalue => V_pred = Z*P_0*Z' + E = 1 - 10 = -9 (non-PD).
    E_bad = reshape([-10.0], 1, 1)
    k_bad = StateSpaceRoutines.KalmanFilter(T, R, C, Q, Z, D, E_bad, s_0, P_0)
    StateSpaceRoutines.update!(k_bad, [0.5]; return_loglh = true)
    @test k_bad.loglh_t == -Inf

    # Positive control: well-posed E => V_pred = 1 + 1 = 2 (PD) => finite loglh,
    # confirming the guard does not over-trigger on healthy draws.
    E_ok = reshape([1.0], 1, 1)
    k_ok = StateSpaceRoutines.KalmanFilter(T, R, C, Q, Z, D, E_ok, s_0, P_0)
    StateSpaceRoutines.update!(k_ok, [0.5]; return_loglh = true)
    @test isfinite(k_ok.loglh_t)
end

# The filter reuses preallocated work arrays (KalmanWorkspace). Its outputs must be
# bitwise equal to the allocating filter, written out here as it was before the work
# arrays (794fb02): one new array per product and period, missing rows dropped per period.
function alloc_filter(regime_inds, y, Ts, Rs, Cs, Qs, Zs, Ds, Es, s_0, P_0)
    Nt, Ns = size(y, 2), length(s_0)
    loglh  = fill(NaN, Nt)
    s_pred = fill(NaN, Ns, Nt); P_pred = fill(NaN, Ns, Ns, Nt)
    s_filt = fill(NaN, Ns, Nt); P_filt = fill(NaN, Ns, Ns, Nt)
    s_t, P_t = s_0, P_0
    for (i, ts) in enumerate(regime_inds)
        T, C = Ts[i], Cs[i]
        RQR′ = Rs[i] * Qs[i] * Rs[i]'
        for t in ts
            s_t = T*s_t + C
            P_t = T*P_t*T' + RQR′
            s_pred[:, t] = s_t; P_pred[:, :, t] = P_t

            y_obs      = y[:, t]
            nonmissing = .![ismissing(x) ? true : isnan(x) for x in y_obs]
            y_obs      = y_obs[nonmissing]
            Z          = Zs[i][nonmissing, :]
            D          = Ds[i][nonmissing]
            E          = Es[i][nonmissing, nonmissing]
            Ny         = length(y_obs)
            y_pred     = Z*s_t + D
            V_pred     = Z*P_t*Z' + E
            V_pred     = (V_pred + V_pred')/2
            V_pred_inv = inv(V_pred)
            dy         = y_obs - y_pred
            PZV        = P_t'*Z'*V_pred_inv
            s_t, P_t   = s_t + PZV*dy, P_t - PZV*Z*P_t
            s_filt[:, t] = s_t; P_filt[:, :, t] = P_t
            loglh[t] = all(real(eigvals(V_pred_inv)) .> 0) == false ? -Inf :
                -(Ny*log(2π) + log(det(V_pred)) + dy'*V_pred_inv*dy)/2
        end
    end
    return loglh, s_pred, P_pred, s_filt, P_filt, s_t, P_t
end

@testset "Kalman Filter (work arrays, bitwise, missing data)" begin
    @load "$path/reference/kalman_filter_args.jld2" y T R C Q Z D E z0 P0
    ym = copy(y)                     # missing data: some series, a whole period, a run of periods
    ym[1, 3:12] .= NaN; ym[2:end, 20] .= NaN; ym[:, 30] .= NaN; ym[end, 40:45] .= NaN
    z0_in, P0_in = copy(z0), copy(P0)
    k = StateSpaceRoutines.KalmanFilter(T, R, C, Q, Z, D, E, z0, P0)
    @test StateSpaceRoutines.kalman_workspace(k, 0.0) isa StateSpaceRoutines.KalmanWorkspace # work arrays in use
    @test StateSpaceRoutines.kalman_workspace(k, 1e-6) === nothing # tol > 0: allocating path
    ref = alloc_filter([1:size(ym, 2)], ym, [T], [R], [C], [Q], [Z], [D], [E], z0, P0)

    out = kalman_filter(ym, T, R, C, Q, Z, D, E, z0, P0)
    for j in 1:5
        @test isequal(out[j], ref[j])
    end
    @test isequal(out[8], ref[6]) && isequal(out[9], ref[7])
    @test isequal(kalman_likelihood(ym, T, R, C, Q, Z, D, E, z0, P0; switching = false), ref[1])
    @test isequal(z0, z0_in) && isequal(P0, P0_in)   # inputs are not overwritten

    # presample: s_0, P_0 are the filtered values at Nt0, and stay so
    out = kalman_filter(ym, T, R, C, Q, Z, D, E, z0, P0; Nt0 = 4)
    @test isequal(out[1], ref[1][5:end]) && isequal(out[4], ref[4][:, 5:end])
    @test isequal(out[6], ref[4][:, 4]) && isequal(out[7], ref[5][:, :, 4])

    # no initial conditions; missing values as `missing`
    s_init, P_init = init_stationary_states(T, R, C, Q)
    ref0 = alloc_filter([1:size(ym, 2)], ym, [T], [R], [C], [Q], [Z], [D], [E], s_init, P_init)
    ymiss = Matrix{Union{Missing, Float64}}(ym); ymiss[isnan.(ym)] .= missing
    @test isequal(kalman_filter(ym, T, R, C, Q, Z, D, E)[1:5], ref0[1:5])
    @test isequal(kalman_filter(ymiss, T, R, C, Q, Z, D, E)[1:5], ref0[1:5])
    @test isequal(kalman_likelihood(ym, T, R, C, Q, Z, D, E; switching = false), ref0[1])

    # more observables than states (P_{t|t} product associated the other way)
    T2, R2, C2, Q2 = [0.9 0.1; 0.0 0.5], [1.0 0.0; 0.3 1.0], [0.1, 0.0], [1.0 0.2; 0.2 0.5]
    Z2, D2, E2     = [1.0 0.0; 0.0 1.0; 0.5 0.5], [0.0, 0.1, 0.2], Matrix(0.1I, 3, 3)
    y2 = ym[:, 1:40]
    ref2 = alloc_filter([1:40], y2, [T2], [R2], [C2], [Q2], [Z2], [D2], [E2], [0.0, 0.0], Matrix(1.0I, 2, 2))
    @test isequal(kalman_filter(y2, T2, R2, C2, Q2, Z2, D2, E2, [0.0, 0.0], Matrix(1.0I, 2, 2))[1:5], ref2[1:5])

    # non-PD V_pred guard through the filter
    @test kalman_likelihood(reshape([0.5], 1, 1), reshape([0.5], 1, 1), reshape([1.0], 1, 1), [0.0],
                            reshape([1.0], 1, 1), reshape([1.0], 1, 1), [0.0], reshape([-10.0], 1, 1),
                            [0.0], reshape([1.0], 1, 1); switching = false) == [-Inf]
end

@testset "Kalman Filter (work arrays, bitwise, regime switching with missing data)" begin
    @load "$path/reference/kalman_filter_args_zlb.jld2" y Ts Rs Cs Qs Zs Ds Es regime_inds
    ym = copy(y)
    ym[1, 3:12] .= NaN; ym[2:end, 20] .= NaN; ym[:, 30] .= NaN
    ym[end, last(regime_inds[1]):(last(regime_inds[1]) + 3)] .= NaN   # across the regime switch
    s_init, P_init = init_stationary_states(Ts[1], Rs[1], Cs[1], Qs[1])
    inds = [1:60, 61:150, 151:size(ym, 2)]                             # three regimes
    for (ri, Tr, Rr, Cr, Qr, Zr, Dr, Er) in
            ((regime_inds, Ts, Rs, Cs, Qs, Zs, Ds, Es),
             (inds, Ts[[1, 2, 1]], Rs[[1, 2, 1]], Cs[[1, 2, 1]], Qs[[1, 2, 1]],
              Zs[[1, 2, 1]], Ds[[1, 2, 1]], Es[[1, 2, 1]]))
        ref = alloc_filter(ri, ym, Tr, Rr, Cr, Qr, Zr, Dr, Er, s_init, P_init)
        out = kalman_filter(ri, ym, Tr, Rr, Cr, Qr, Zr, Dr, Er)
        for j in 1:5
            @test isequal(out[j], ref[j])
        end
        @test isequal(out[8], ref[6]) && isequal(out[9], ref[7])
        @test isequal(kalman_likelihood(ri, ym, Tr, Rr, Cr, Qr, Zr, Dr, Er), ref[1])
        @test isequal(kalman_likelihood(ri, ym, Tr, Rr, Cr, Qr, Zr, Dr, Er, s_init, P_init), ref[1])
    end
end


nothing
