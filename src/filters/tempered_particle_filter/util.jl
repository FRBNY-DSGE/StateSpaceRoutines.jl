#=macro mypar(parallel, ex)
    return :( $(esc(parallel)) ? (@sync @distributed $(esc(ex))) : $(esc(ex)) )
end
=#
function bisection(f::Function, a::Number, b::Number;
                   xtol::AbstractFloat = 1e-1, maxiter::Int = 1000)
    fa = f(a)
    sign(fa) == sign(f(b)) && throw("No real root in [a, b]")
    i = 0
    c = 0.0
    while b-a > xtol
        i += 1
        i == maxiter && throw("Max number of iterations exceeded")
        c = (a+b)/2
        fc = f(c)
        if fc ≈ 0
            break
        elseif sign(fa) == sign(fc)
            # Root is in the right half of [a, b]
            a = c
            fa = fc
        else
            # Root is in the left half of [a, b]
            b = c
        end
    end
    return c
end

function fast_mvnormal_pdf(x::Vector{Float64})
    coeff_term = (2*pi)^(-length(x)/2)
    exp_term   = exp(-(1/2) * dot(x, x))
    return coeff_term*exp_term
end

function fast_mvnormal_pdf(x::Vector{Float64}, det_Σ::Float64, inv_Σ::Matrix{Float64})
    coeff_term = (2*pi)^(-length(x)/2) * det_Σ^(-1/2)
    exp_term   = exp(-(1/2) * dot(x, inv_Σ * x))
    return coeff_term*exp_term
end