using Images, Plots, Zygote, FiniteDifferences
using LinearAlgebra, Random, Statistics, Printf

function Rosenbrock(x, y; a=1, b=100)
    return (a .- x) .^ 2 + b .* (y - x .^ 2) .^ 2
end

function Himmelblau(x, y)
    return (x .^ 2 + y .- 11) .^ 2 + (x + y .^ 2 .- 7) .^ 2
end

function plotFct(fct, xul, yul, resolution; save=false, filename="name.png")

    x = LinRange(xul[1], xul[2], resolution)
    x = repeat(x, 1, resolution)
    y = LinRange(yul[1], yul[2], resolution)
    y = repeat(y, 1, resolution)
    result = fct(transpose(x), y)

    z = ((2 .^ (1.0:0.01:4.0)) .- 2) .^ 3
    heatmap(result, color=cgrad(:haline, z, scale=exp),
            xticks=([1, 250, 500, 750, 1000],
                    convert(Array{Int64}, LinRange(xul[1], xul[2], 5))),
            yticks=([1, 250, 500, 750, 1000],
                    convert(Array{Int64}, LinRange(yul[1], yul[2], 5))),
            aspect_ratio=:equal, size=(500, 500),
            background_color = :transparent, foreground_color=:black)

    if save
        savefig("C:/Users/mo_-_/TorchProjects/NlOptJulia/Plots/" * filename)
    end
end

function gradFD(f, x, y)
    epsilon = sqrt(eps(eltype(x)))
    gradsx = zero(x)
    gradsy = zero(y)
    gradsx .= (f(x .+ epsilon, y) - f(x .- epsilon, y)) / (2 * epsilon)
    gradsy .= (f(x, y .+ epsilon) - f(x, y .- epsilon)) / (2 * epsilon)
    return (gradsx, gradsy)
end

function gradAD(f, x, y)
    return gradient((x, y) -> sum(f(x, y)), x, y)
end

function gradientDescent(f, getGrad1, getGrad2, x, y; steps=2000, lr=0.01)

    x1 = deepcopy(x)
    x2 = deepcopy(x)
    y1 = deepcopy(y)
    y2 = deepcopy(y)
    meanErrNorm = []

    for t = 1:steps
        σ = lr / sqrt(t)

        grad1 = getGrad1(f, x1, y1)
        x1 -= σ .* grad1[1]
        y1 -= σ .* grad1[2]

        grad2 = getGrad2(f, x2, y2)
        x2 -= σ .* grad2[1]
        y2 -= σ .* grad2[2]
        push!(meanErrNorm, mean(f(x1, y1) - f(x2, y2)) ^ 2)
    end

    return meanErrNorm, sum(sqrt.((x1 - x2) .^ 2 + (y1 - y2) .^ 2) .> 2)
end


#plotFct(Himmelblau, (-6.0, 6.0), (-6.0, 6.0), 1000,
#        save=false, filename="himmelblau.png")
#plotFct(Rosenbrock, (-2.0, 2.0), (-1.0, 3.0), 1000,
#        save=false, filename="rosenbrock.png")

function getErrors(f, xul, yul, arrType; resolution=100)
    xul = convert(Vector{arrType}, xul)
    yul = convert(Vector{arrType}, yul)

    x = LinRange(xul[1], xul[2], resolution)
    x = transpose(repeat(x, 1, resolution))[:]
    y = LinRange(yul[1], yul[2], resolution)
    y = repeat(y, 1, resolution)[:]

    return gradientDescent(Himmelblau, gradAD, gradFD, x, y)
end

xul = [-2.0, 2.0]# .- 0.270845
yul = [-1.0, 3.0]# .- 0.923039

errors16, wrongMin16 = getErrors(Rosenbrock, xul, yul, Float16)
errors32, wrongMin32 = getErrors(Rosenbrock, xul, yul, Float32)
errors64, wrongMin64 = getErrors(Rosenbrock, xul, yul, Float64)

print(wrongMin16, ", ", wrongMin32, ", ", wrongMin64)

plot(range(1, 2000, length=2000), [errors16, errors32, errors64],
    xlabel="Iteration", ylabel="Mean distance", ylims=(1e-30, 1.0),
    legend=:topright, label=["16 bit float" "32 bit float" "64 bit float"],
    background_color = :transparent, foreground_color=:black)
plot!(yscale=:log10)
hline!([maximum(errors16), maximum(errors32), maximum(errors64)],
      color=:gray, linestyle=:dash, label="")

savefig("C:/Users/mo_-_/TorchProjects/NlOptJulia/Plots/" * "RosenbrockDist")