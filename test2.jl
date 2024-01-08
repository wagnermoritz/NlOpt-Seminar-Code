using Images, Plots, Printf, LinearAlgebra
include("/home/m0e/Documents/NlOpt-Seminar-Code/DiffStuff.jl")
using .ReverseAD: Variable, backward!, grad, +, -, *, /, ^, convert
using .ForwardFD: gradFD


function Rosenbrock(x, y; a=1, b=100)
    return (a .- x) .^ 2 + b .* (y - x .^ 2) .^ 2
end

a = [1.0, 2.0, -3.0]
b = [5.1, 3.9, 11.0]

x = convert.(Variable, a)
y = convert.(Variable, b)
z = Rosenbrock(x, y)
backward!.(z)
print(grad.(x), grad.(y))

print(gradFD(Rosenbrock, a, b))
