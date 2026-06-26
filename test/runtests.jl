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

# Common imports seeded into each per-file module below (mirrors what this file's
# top-level `using`s provided when every test was included into a shared `Main`).
const _prelude = :(using StateSpaceRoutines, Test, HDF5, JLD2, FileIO,
                         LinearAlgebra, PDMats, Distributions, Random)

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
