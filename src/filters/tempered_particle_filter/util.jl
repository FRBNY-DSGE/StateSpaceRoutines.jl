"""
```
update_c!(c_in::Float64, accept_in::Float64, target_in::Float64)
```
Updates value of c by expression that is function of the target and mean acceptance rates.
Returns the new c, in addition to storing it in the model settings.

"""
@inline function update_c!(c_in::Float64, accept_in::Float64, target_in::Float64)
    c_out = c_in*(0.95 + 0.1*exp(20*(accept_in - target_in))/(1 + exp(20*(accept_in - target_in))))
    return c_out
end

function bisection(f::Function, a::Number, b::Number; tol::AbstractFloat=1e-1, maxiter::Integer=1000)
    fa = f(a)
    fa * f(b) <= 0 || throw("No real root in [a,b]")
    i = 0
    c = 0
    while b-a > tol
        i += 1
        i != maxiter || throw("Max iteration exceeded")
        c = (a+b)/2
        fc = f(c)
        if fc ≈ 0
            break
        elseif sign(fa) == sign(fc)
            a = c
            fa = fc # Root is in the right half of [a,b]
        else
            b = c # Root is in the left half of [a,b]
        end
    end
    return c
end

function fast_mvnormal_pdf(x::Vector{Float64}, μ::Vector{Float64}, detΣ::Float64, invΣ::Matrix{Float64})
    coeff_term = (2*pi)^(-length(x)/2) * detΣ^(-1/2)
    exp_term   = exp(-(1/2) * dot((x - μ), invΣ*(x - μ)))
    return coeff_term*exp_term
end

# The return type of reduce functions must be the same type as the tuple of arguments being input
# E.g. If args is a tuple of Vector{Float64}, then the return argument must also be a Vector{Float64}
# Thus, to implement a scalar reduce function, where each individual iteration returns
# n scalars, and we want the output to be reduced to n vectors, where the i-th vector
# contains all of the i-th scalars from each individual iteration, then we must modify the
# individual iterations to return n singleton vectors (one element vectors) of those n
# scalars so as to preserve the homogeneity of the input/output type coming into/out of
# the scalar reduce function.
# This would not work if the input argument types were just Ints or Float64s
# since a return type of Int/Float64 for scalar reduce function does not permit that
# function to return collections of items (because an Int/Float64 can only contain a
# single value).
# E.g.
# a, b = @parallel (scalar_reduce) for i in 1:10000
    # [[1], [2]]
# end
# a = [1, 1, 1, ...]
# b = [2, 2, 2, ...]

# Input/Output type: Vector{Vector{Float64}}
function scalar_reduce(args...)
    return_arg = args[1]
    for (i, arg) in enumerate(args[2:end])
        for (j, el) in enumerate(arg)
            append!(return_arg[j], el)
        end
    end
    return return_arg
end

# Same logic applies to the vector reduce, where each individual iteration returns n
# Vector types, and we want to vector reduce to n matrices, where the i-th column of that
# matrix corresponds to the i-th vector from an individual iteration.
# Input/Output type: Vector{Matrix{Float64}}
function vector_reduce(args...)
    nargs1 = length(args) # The number of times the loop is run
    nargs2 = length(args[1]) # The number of variables output by a single run

    return_arg = args[1]
    for i in 1:nargs2
        for j in 2:nargs1
            return_arg[i] = hcat(return_arg[i], args[j][i])
        end
    end
    return return_arg
end

# The following two functions are to ensure type conformity of the return arguments
function scalar_reshape(args...)
    n_args = length(args)
    return_arg = Vector{Vector{Float64}}(n_args)
    for i in 1:n_args
        arg = typeof(args[i]) <: Vector ? args[i] : [args[i]]
        return_arg[i] = arg
    end
    return return_arg
end

function vector_reshape(args...)
    n_args = length(args)
    return_arg = Vector{Matrix{Float64}}(n_args)
    for i in 1:n_args
        arg = typeof(args[i]) <: Vector ? args[i] : [args[i]]
        return_arg[i] = reshape(arg, length(arg), 1)
    end
    return return_arg
end
