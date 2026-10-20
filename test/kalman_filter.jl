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

@testset "Kalman likelihood split regime uses existing systems without mutation" begin
    inds = UnitRange{Int}[1:4, 5:8, 9:12]
    obs = reshape(collect(0.1:0.1:1.2), 1, :)
    Ts = [reshape([v], 1, 1) for v in (0.5, 0.6, 0.7)]
    Rs = [reshape([1.0], 1, 1) for _ in 1:3]
    Cs = [zeros(1) for _ in 1:3]
    Qs = [reshape([1.0], 1, 1) for _ in 1:3]
    Zs = [reshape([1.0], 1, 1) for _ in 1:3]
    Ds = [zeros(1) for _ in 1:3]
    Es = [reshape([0.1], 1, 1) for _ in 1:3]
    baseline = kalman_likelihood(inds, obs, Ts, Rs, Cs, Qs, Zs, Ds, Es)
    filt = kalman_filter(inds, obs, Ts, Rs, Cs, Qs, Zs, Ds, Es)

    for d in (2, 6, 10)
        input_inds = copy(inds)
        split_lh, split_state = kalman_likelihood(input_inds, obs, Ts, Rs, Cs, Qs,
            Zs, Ds, Es; add_zlb_duration=(true, d))
        @test isequal(split_lh, baseline)
        @test isequal(split_state, filt[4][:, d])
        @test input_inds == inds
    end

    duplicate(v) = vcat(v[1:2], v[2:2], v[3:3])
    legacy_inds = copy(inds)
    legacy_lh, legacy_state = kalman_likelihood(legacy_inds, obs,
        duplicate(Ts), duplicate(Rs), duplicate(Cs), duplicate(Qs),
        duplicate(Zs), duplicate(Ds), duplicate(Es); add_zlb_duration=(true, 6))
    @test isequal(legacy_lh, baseline)
    @test isequal(legacy_state, filt[4][:, 6])
    @test legacy_inds == inds

    input_inds = copy(inds)
    @test_throws BoundsError kalman_likelihood(input_inds, obs, Ts, Rs, Cs, Qs,
        Zs, Ds, Es; add_zlb_duration=(true, 13))
    @test input_inds == inds
    input_inds = copy(inds)
    @test_throws BoundsError kalman_likelihood(input_inds, obs, Ts[1:2], Rs, Cs, Qs,
        Zs, Ds, Es; add_zlb_duration=(true, 6))
    @test input_inds == inds
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


nothing
