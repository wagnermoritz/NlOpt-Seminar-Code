using Images, Plots, Printf, LinearAlgebra
include("./DiffStuff.jl")
using .ReverseAD: Variable, backward!, grad, +, -, *, /, ^, convert
using .FiniteDiff: centralGrad, forwardGrad
using .ComplexStep: complexGrad


function Himmelblau(x, y)
    return (x .^ 2 + y .- 11) .^ 2 + (x + y .^ 2 .- 7) .^ 2
end

function grad_Himmelblau(x, y)
    return 2 * (-7 .+ x + y .^ 2 + 2 * x .* (-11 .+ x .^ 2 + y)),
           2 * (-11 .+ x .^ 2 + y + 2 * y .* (-7 .+ x + y .^ 2))
end


# compute the error of the gradient of the function f approximated by :get_grad:
# compared to the exact gradient :grad_f:. The error is evaluated on a grid
# with upper and lower bounds on x and y given in xul and yul.
function getErrorsFD(f, grad_f, get_grad, xul, yul, arrType, epsul; resolution=100)
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
        distx = abs.(dx - true_dx)
        disty = abs.(dy - true_dy)
        push!(errors, (sum(distx) + sum(disty)) / (2 * resolution ^ 2))
    end

    return errors
end

# compute the error of the gradient of the function f computed by reverse AD
# compared to the exact gradient :grad_f:. The error is evaluated on a grid
# with upper and lower bounds on x and y given in xul and yul.
function getErrorAD(f, grad_f, xul, yul, arrType; resolution=100)

    xul = convert(Vector{arrType}, xul)
    yul = convert(Vector{arrType}, yul)
    x = LinRange(xul[1], xul[2], resolution)
    x = transpose(repeat(x, 1, resolution))[:]
    y = LinRange(yul[1], yul[2], resolution)
    y = repeat(y, 1, resolution)[:]

    true_dx, true_dy = grad_f(x, y)

    x = convert.(Variable, x)
    y = convert.(Variable, y)
    res = f(x, y)
    backward!.(res)
    dx = grad.(x)
    dy = grad.(y)
    distx = abs.(dx - true_dx)
    disty = abs.(dy - true_dy)

    return (sum(distx) + sum(disty)) / (2 * resolution ^ 2)
end


xul = [-6.0, 6.0]
yul = [-6.0, 6.0]
epsul = [-16, 1]

errors32f = getErrorsFD(Himmelblau, grad_Himmelblau, forwardGrad, xul, yul, Float32, epsul)
errors64f = getErrorsFD(Himmelblau, grad_Himmelblau, forwardGrad, xul, yul, Float64, epsul)
errors32c = getErrorsFD(Himmelblau, grad_Himmelblau, centralGrad, xul, yul, Float32, epsul)
errors64c = getErrorsFD(Himmelblau, grad_Himmelblau, centralGrad, xul, yul, Float64, epsul)

plot(10.0 .^ LinRange(epsul[1], epsul[2], 1000), 
     [errors32f, errors64f, errors32c, errors64c],
     xlabel="ε", ylabel="mean error", legend=:bottomleft,
     label=["one-sided 32 bit" "one-sided 64 bit" "central 32 bit" "central 64 bit"],
     background_color=:white, foreground_color=:black, ylims=[10^-9, 10^3])
plot!(10.0 .^ LinRange(epsul[1], epsul[2], 1000), 10.0 .^ LinRange(epsul[1], epsul[2], 1000),
      label="ε", color=:black, linestyle=:dash)
plot!(10.0 .^ LinRange(epsul[1], epsul[2], 1000), (10.0 .^ LinRange(epsul[1], epsul[2], 1000)) .^ 2,
      label="ε^2", color=:black)
plot!(xscale=:log10, yscale=:log10)

vline!([eps(eltype(errors32f))^(1/2)], color=1, linestyle=:dash, label="")
vline!([eps(eltype(errors64f))^(1/2)], color=2, linestyle=:dash, label="")
vline!([eps(eltype(errors32c))^(1/3)], color=3, linestyle=:dash, label="")
vline!([eps(eltype(errors64c))^(1/3)], color=4, linestyle=:dash, label="")

savefig("./Plots/" * "Forward vs Central")

errors32i = getErrorsFD(Himmelblau, grad_Himmelblau, complexGrad, xul, yul, Float32, epsul)
errors64i = getErrorsFD(Himmelblau, grad_Himmelblau, complexGrad, xul, yul, Float64, epsul)
errorAD32 = getErrorAD(Himmelblau, grad_Himmelblau, xul, yul, Float32)
errorAD64 = getErrorAD(Himmelblau, grad_Himmelblau, xul, yul, Float64)

plot(10.0 .^ LinRange(epsul[1], epsul[2], 1000), 
     [errors32i, errors64i, errors32c, errors64c],
     xlabel="ε", ylabel="mean error", legend=:bottomleft,
     label=["complex 32 bit" "complex 64 bit" "central 32 bit" "central 64 bit"],
     background_color=:white, foreground_color=:black, ylims=[10^-16, 10^3])
plot!(xscale=:log10, yscale=:log10)

hline!([errorAD32], color=:black, linestyle=:dash, label="automatic 32 bit")
hline!([errorAD64], color=:black, label="automatic 64 bit")

savefig("./Plots/" * "Complex vs Central")
