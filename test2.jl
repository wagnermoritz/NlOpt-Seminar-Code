using Images, Plots, Printf, LinearAlgebra
include("/home/mo/Documents/Code/NlOpt-Seminar-Code/DiffStuff.jl")
using .ReverseAD: Variable, backward!, grad, +, -, *, /, ^, convert
using .FiniteDiff: centralGrad, forwardGrad


function Rosenbrock(x, y)
    return (1 .- x) .^ 2 + 100 * (y - x .^ 2) .^ 2
end

function grad_Rosenbrock(x, y)
    return 2 * (-1 .+ x + 200 * x .^ 3 - 200 .* x .* y),
           200 * (-x .^ 2 + y)
end

function Himmelblau(x, y)
    return (x .^ 2 + y .- 11) .^ 2 + (x + y .^ 2 .- 7) .^ 2
end

function grad_Himmelblau(x, y)
    return 2 * (-7 .+ x + y .^ 2 + 2 * x .* (-11 .+ x .^ 2 + y)),
           2 * (-11 .+ x .^ 2 + y + 2 * y .* (-7 .+ x + y .^ 2))
end


function getErrors(f, grad_f, get_grad, xul, yul, arrType, epsul; resolution=100)

    xul = convert(Vector{arrType}, xul)
    yul = convert(Vector{arrType}, yul)
    x = LinRange(xul[1], xul[2], resolution)
    x = transpose(repeat(x, 1, resolution))[:]
    y = LinRange(yul[1], yul[2], resolution)
    y = repeat(y, 1, resolution)[:]

    epsrange = convert(Vector{arrType}, 10.0 .^ LinRange(epsul[1], epsul[2], 1000))
    errors = arrType[]

    for eps in epsrange
        dx, dy = get_grad(f, x, y; epsilon=eps)
        true_dx, true_dy = grad_f(x, y)
        distx = dx - true_dx
        disty = dy - true_dy
        push!(errors, sum(sqrt.(distx .^ 2 + disty .^ 2)) / (resolution ^ 2))
    end

    return errors
end


xul = [-6.0, 6.0]# .- 0.270845
yul = [-6.0, 6.0]# .- 0.923039
epsul = [-16, 1]

errors32f = getErrors(Himmelblau, grad_Himmelblau, forwardGrad, xul, yul, Float32, epsul)
errors64f = getErrors(Himmelblau, grad_Himmelblau, forwardGrad, xul, yul, Float64, epsul)
errors32c = getErrors(Himmelblau, grad_Himmelblau, centralGrad, xul, yul, Float32, epsul)
errors64c = getErrors(Himmelblau, grad_Himmelblau, centralGrad, xul, yul, Float64, epsul)

plot(10.0 .^ LinRange(epsul[1], epsul[2], 1000), 
     [errors32f, errors64f, errors32c, errors64c],
     xlabel="epsilon", ylabel="mean error", legend=:bottomleft,
     label=["forward 32 bit" "forward 64 bit" "central 32 bit" "central 64 bit"],
     background_color=:white, foreground_color=:black)
plot!(xscale=:log10, yscale=:log10)

vline!([eps(eltype(errors32f))^(1/2)], color=1, linestyle=:dash, label="")
vline!([eps(eltype(errors64f))^(1/2)], color=2, linestyle=:dash, label="")
vline!([eps(eltype(errors32c))^(1/3)], color=3, linestyle=:dash, label="")
vline!([eps(eltype(errors64c))^(1/3)], color=4, linestyle=:dash, label="")






# a = [1.0, 2.0, -3.0]
# b = [5.1, 3.9, 11.0]

# x = convert.(Variable, a)
# y = convert.(Variable, b)
# z = Himmelblau(x, y)
# backward!.(z)
# print(grad.(x), grad.(y))
# print("\n")
# dx, dy = gradFD(Himmelblau, a, b)
# print(dx, dy)
# print("\n")
# print(grad_Himmelblau(a, b))
