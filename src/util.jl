# File copied from SMC.jl
"""
```
function scalar_reduce(args...)
```
Each individual iteration returns n scalars. The output is reduced to n vectors,
where the i-th vector contains all of the i-th scalars from each iteration.

The return type of reduce functions must be the same type as the tuple of
arguments passed in. If args is a tuple of Vector{Float64}, then the return
argument will be a Vector{Float64}.

e.g.
a, b = @parallel (scalar_reduce) for i in 1:10000
           [[1], [2]]
       end
a = [1, 1, 1, ...]
b = [2, 2, 2, ...]

Input/Output type: Vector{Vector{Float64}}
"""
function scalar_reduce(args...)
    return_arg = args[1]
    for (i, arg) in enumerate(args[2:end])
        for (j, el) in enumerate(arg)
            append!(return_arg[j], el)
        end
    end
    return return_arg
end

"""
```
function vector_reduce(args...)
```
Each individual iteration returns n Vector types; we vector-reduce to n matrices, where
the i-th column of that matrix corresponds to the i-th vector from an individual iteration.

Input/Output type: Vector{Matrix{Float64}}
"""
function vector_reduce(args...)
    nargs1 = length(args)    # The number of times the loop is run
    nargs2 = length(args[1]) # The number of variables output by a single run

    return_arg = args[1]
    for i in 1:nargs2
        for j in 2:nargs1
            return_arg[i] = hcat(return_arg[i], args[j][i])
        end
    end
    return return_arg
end

"""
```
function vec_red_scal(args...)
```
vector_reduce but scalar_reduce for last argument
Hard coded for 3 arguments
"""
function vec_scal_reduce(args...)
    nargs1 = length(args)    # The number of times the loop is run
    nargs2 = length(args[1]) # The number of variables output by a single run

    return_arg = args[1]
    for i in 1:nargs2
        for j in 2:nargs1
            if i == nargs2
                append!(return_arg[i], args[j][i][1])
                #return_arg[i] = vcat(return_arg[i], args[j][i])
            else
                return_arg[i] = hcat(return_arg[i], args[j][i])
            end
        end
    end
    return return_arg
#=
    nargs1 = length(args)    # The number of times the loop is run
    nargs2 = length(args[1]) # The number of variables output by a single run

    if nargs2 == 1
        return args
    end

    arg1, arg2, arg3 = args[1]
    #arg2 = args[1][2]
    #arg3 = args[1][3]

    for j in 2:nargs1
        arg1 = hcat(arg1, args[j][1])
        arg2 = hcat(arg2, args[j][2])
        arg3 = vcat(arg3, args[j][3])
    end

    return_arg = (arg1, arg2, arg3)
    return return_arg
=#
end

"""
```
function scalar_reshape(args...)
```
Function ensures type conformity of the return arguments.
"""
function scalar_reshape(args...)
    n_args = length(args)
    return_arg = Vector{Vector{Float64}}(undef, n_args)
    for i in 1:n_args
        arg = typeof(args[i]) <: Vector ? args[i] : [args[i]]
        return_arg[i] = arg
    end
    return return_arg
end

"""
```
function vector_reshape(args...)
```
Function ensures type conformity of the return arguments.
"""
function vector_reshape(args...)
    n_args = length(args)
    return_arg = Vector{Matrix{Float64}}(undef, n_args)
    for i in 1:n_args
        arg = typeof(args[i]) <: Vector ? args[i] : [args[i]]
        return_arg[i] = reshape(arg, length(arg), 1)
    end
    return return_arg
end

"""
```
sendto(p::Int; args...)
```
Function to send data from master process to particular worker, p.
Code from ChrisRackauckas, avavailable at:
 https://github.com/ChrisRackauckas/ParallelDataTransfer.jl/blob/master/src/ParallelDataTransfer.jl.
"""
function sendto(p::Int; args...)
    for (nm, val) in args
        @spawnat(p, Core.eval(Main, Expr(:(=), nm, val)))
    end
end

"""
```
sendto(ps::AbstractVector{Int}; args...)
```
Function to send data from master process to list of workers.
Code from ChrisRackauckas, available at:
https://github.com/ChrisRackauckas/ParallelDataTransfer.jl/blob/master/src/ParallelDataTransfer.jl.
"""
function sendto(ps::AbstractVector{Int}; args...)
    for p in ps
        sendto(p; args...)
    end
end
