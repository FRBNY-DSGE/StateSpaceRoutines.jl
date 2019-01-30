path = dirname(@__FILE__)
file = h5open("$path/reference/kalman_filter_args.h5", "r")
y = read(file, "data")

T, R, C    = read(file, "TTT"), read(file, "RRR"), read(file, "CCC")
Q, Z, D, E = read(file, "QQ"), read(file, "ZZ"), read(file, "DD"), read(file, "EE")
s_0, P_0   = read(file, "z0"), read(file, "P0")
close(file)

# Basic Chand Recursion (all arguments, no presample)
out = chand_recursion(y, T, R, C, Q, Z, D, E, s_0, P_0, allout = true)
@testset "Basic Chand Recursion (no presample, all)" begin
    h5open("$path/reference/kalman_filter_out.h5", "r") do h5
        @test read(h5, "log_likelihood") ≈ out[1]
        @test read(h5, "marginal_loglh") ≈ out[2]
        @test read(h5, "s_TT") ≈ out[3]
        @test read(h5, "P_TT") ≈ out[4]
    end
end

# Chand Recursion (no initial conditions)
out = chand_recursion(y, T, R, C, Q, Z, D, E, allout = true)
@testset "Chand Recursion (no initial conditions)" begin
      h5open("$path/reference/kalman_filter_out.h5", "r") do h5
          @test read(h5, "log_likelihood") ≈ out[1]
          @test read(h5, "marginal_loglh") ≈ out[2]
          @test read(h5, "s_TT") ≈ out[3]
          @test read(h5, "P_TT") ≈ out[4]
      end
  end

# Chand Recursion (presample)
out = chand_recursion(y, T, R, C, Q, Z, D, E, Nt0 = 4, allout = true)
@testset "Chand Recursion (presample)" begin
    h5open("$path/reference/kalman_filter_out_presample.h5", "r") do h5
        @test read(h5, "log_likelihood") ≈ out[1]
        @test read(h5, "marginal_loglh") ≈ out[2]
        @test read(h5, "s_TT") ≈ out[3]
        @test read(h5, "P_TT") ≈ out[4]
    end
end


# Basic Chand Recursion (all arguments, no presample)
out = chand_recursion(y, T, R, C, Q, Z, D, E, s_0, P_0, allout = true, tol = 1e-4)
@testset "Basic Chand Recursion (no presample, all)" begin
    h5open("$path/reference/kalman_filter_out.h5", "r") do h5
        @test read(h5, "log_likelihood") ≈ out[1]
        @test read(h5, "marginal_loglh") ≈ out[2]
        @test read(h5, "s_TT") ≈ out[3]
        @test read(h5, "P_TT") ≈ out[4]
    end
end

# Chand Recursion (no initial conditions)
out = chand_recursion(y, T, R, C, Q, Z, D, E, allout = true, tol = 1e-4)
@testset "Chand Recursion (no initial conditions)" begin
      h5open("$path/reference/kalman_filter_out.h5", "r") do h5
          @test read(h5, "log_likelihood") ≈ out[1]
          @test read(h5, "marginal_loglh") ≈ out[2]
          @test read(h5, "s_TT") ≈ out[3]
          @test read(h5, "P_TT") ≈ out[4]
      end
  end

# Chand Recursion (presample)
out = chand_recursion(y, T, R, C, Q, Z, D, E, Nt0 = 4, allout = true, tol = 1e-4)
@testset "Chand Recursion (presample)" begin
    h5open("$path/reference/kalman_filter_out_presample.h5", "r") do h5
        @test read(h5, "log_likelihood") ≈ out[1]
        @test read(h5, "marginal_loglh") ≈ out[2]
        @test read(h5, "s_TT") ≈ out[3]
        @test read(h5, "P_TT") ≈ out[4]
    end
end

