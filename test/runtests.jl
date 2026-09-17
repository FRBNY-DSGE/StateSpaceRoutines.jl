using StateSpaceRoutines
using Test, HDF5, JLD2, FileIO, LinearAlgebra, PDMats, Distributions, Random

my_tests = [
            "kalman_filter",
            "chand_recursion",
            "tempered_particle_filter",
            "smoothers",
            "dyn_measure_tpf",
            "tv_tpf",
            "initialization",
            "util",
            "filters/util",
            "smoothers/util",
            "filters/tempered_particle_filter/util",
            "EnKF",
            "parallel_tempered_particle_filter"
            ]

# Common imports + helpers seeded into each per-file module below (mirrors what this
# file's top-level `using`s provided when every test was included into a shared `Main`).
const _prelude = quote
    using StateSpaceRoutines, Test, HDF5, JLD2, FileIO,
          LinearAlgebra, PDMats, Distributions, Random

    # JLD2 may deserialize a saved MvNormal as an opaque JLD2.ReconstructedMutable when the
    # installed Distributions/PDMats type layout differs from when the reference file was
    # written (CI commonly resolves newer versions than the machine that saved it). Rebuild
    # a real MvNormal from the loaded fields so cov()/indexing/the filter work regardless.
    function as_mvnormal(F)
        F isa Distributions.MvNormal && return F
        μ = collect(getproperty(F, :μ))
        Σobj = getproperty(F, :Σ)
        Σ = if Σobj isa AbstractMatrix
            Matrix(Σobj)                              # already a (PD)Matrix
        elseif hasproperty(Σobj, :mat)
            Matrix(getproperty(Σobj, :mat))           # PDMat.mat field
        else
            Matrix(getproperty(Σobj, :chol))          # fall back to the Cholesky factor
        end
        return MvNormal(μ, Σ)
    end
end

# The parallel test runs directly in Main (see below); give Main the same helper.
eval(_prelude)

# Run each test file in its OWN module. Several files define top-level helpers with
# the same names (Φ, Ψ, Ψt, …); in a shared `Main` those redefinitions triggered
# noisy "method overwritten" warnings. Isolating per file removes them. Each file
# also brings its own `using`s, so the prelude only needs the common set.
for test in my_tests
    @testset "$test.jl" begin
        if test == "parallel_tempered_particle_filter"
            # This file is @everywhere-heavy, and @everywhere always evaluates in
            # Main — so its worker-side definitions and an isolated module's local
            # definitions would split-brain (the local Ψ never gets overwritten).
            # Run it directly in Main; it's the last test, and every other file is
            # isolated, so nothing else pollutes Main to collide with it.
            include(joinpath(@__DIR__, "$test.jl"))
        else
            m = Module(Symbol("Test_", replace(test, "/" => "_")))
            Core.eval(m, _prelude)
            Base.include(m, joinpath(@__DIR__, "$test.jl"))
        end
    end
end
