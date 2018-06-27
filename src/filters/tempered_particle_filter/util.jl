macro mypar(parallel, ex)
    return :( $(esc(parallel)) ? (@sync @parallel $(esc(ex))) : $(esc(ex)) )
end

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

function bisection(f::Function, a::Number, b::Number; xtol::AbstractFloat=1e-1, maxiter::Integer=1000)
    fa = f(a)
    fa * f(b) <= 0 || throw("No real root in [a,b]")
    i = 0
    c = 0.0
    while b-a > xtol
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

function fast_mvnormal_pdf(x::Vector{Float64})
    coeff_term = (2*pi)^(-length(x)/2)
    exp_term   = exp(-(1/2) * dot(x, x))
    return coeff_term*exp_term
end

function fast_mvnormal_pdf(x::Vector{Float64}, detΣ::Float64, invΣ::Matrix{Float64})
    coeff_term = (2*pi)^(-length(x)/2) * detΣ^(-1/2)
    exp_term   = exp(-(1/2) * dot(x, invΣ * x))
    return coeff_term*exp_term
end