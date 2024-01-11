module ReverseAD

# struct holding a value and all information needed for reverse AD
mutable struct Variable{T} <: Number
    value::T
    adjoint::T
    pjis::Vector{T}
    parents::Vector{Variable{T}}

    # for checking if the partial derivative is finished
    numchildren::Integer
    contributionfrom::Integer

    function Variable(value)
        x = new{typeof(value)}()
        x.value = value
        x.adjoint = zero(value)
        x.pjis = typeof(value)[]
        x.parents = typeof(x)[]
        x.numchildren = 0
        x.contributionfrom = 0
        return x
    end
end

# set all adjoints except the adjoint of x to zero
function zero_adjoints!(x::Variable)
    for i in 1:length(x.parents)
        x.parents[i].adjoint = zero(x.parents[i].adjoint)
        x.parents[i].contributionfrom = 0
        zero_adjoints!(x.parents[i])
    end
    return
end

# accumulate derivatives in all adjoints
function recursive_backward!(x::Variable)
    for i in 1:length(x.parents)
        x.parents[i].adjoint += x.pjis[i] * x.adjoint
        x.parents[i].contributionfrom += 1
        if x.parents[i].numchildren == x.parents[i].contributionfrom
            recursive_backward!(x.parents[i])
        end
    end
    return
end

# reverse sweep
function backward!(x::Variable)
    x.adjoint = one(x.adjoint)
    x.contributionfrom = x.numchildren
    zero_adjoints!(x)
    recursive_backward!(x)
end

function grad(x::Variable)
    x.adjoint
end

# overload operators to build up reverse AD graph
import Base: +, -, *, /, ^, convert

function +(x::Variable, y::Variable)
    x.numchildren += 1
    y.numchildren += 1
    z = Variable(x.value + y.value)
    z.pjis = [one(x.value), one(y.value)]
    z.parents = [x, y]
    return z
end

function +(x::Variable, y::Real)
    x.numchildren += 1
    z = Variable(x.value + y)
    z.pjis = [one(x.value)]
    z.parents = [x]
    return z
end

function +(x::Real, y::Variable)
    return y + x
end

function -(x::Variable, y::Variable)
    x.numchildren += 1
    y.numchildren += 1
    z = Variable(x.value - y.value)
    z.pjis = [one(x.value), -one(y.value)]
    z.parents = [x, y]
    return z
end

function -(x::Variable, y::Real)
    x.numchildren += 1
    z = Variable(x.value - y)
    z.pjis = [one(x.value)]
    z.parents = [x]
    return z
end

function -(x::Real, y::Variable)
    return -y + x
end

function -(x::Variable)
    x.numchildren += 1
    z = Variable(-x.value)
    z.pjis = [-one(x.value)]
    z.parents = [x]
    return z
end

function *(x::Variable, y::Variable)
    x.numchildren += 1
    y.numchildren += 1
    z = Variable(x.value * y.value)
    z.pjis = [y.value, x.value]
    z.parents = [x, y]
    return z
end

function *(x::Variable, y::Real)
    x.numchildren += 1
    z = Variable(x.value * y)
    z.pjis = [y]
    z.parents = [x]
    return z
end

function *(x::Real, y::Variable)
    return y * x
end

function /(x::Variable, y::Variable)
    x.numchildren += 1
    y.numchildren += 1
    z = Variable(x.value / y.value)
    z.pjis = [1 / y.value, -x.value / (y.value ^ 2)]
    z.parents = [x, y]
    return z
end

function /(x::Variable, y::Real)
    x.numchildren += 1
    z = Variable(x.value / y)
    z.pjis = [1 / y]
    z.parents = [x]
    return z
end

function /(x::Real, y::Variable)
    y.numchildren += 1
    z = Variable(x / y.value)
    z.pjis = [-x / (y.value ^ 2)]
    z.parents = [y]
    return z
end

function ^(x::Variable, y::AbstractFloat)
    x.numchildren += 1
    z = Variable(x.value ^ y)
    z.pjis = [y * x.value ^ (y - 1)]
    z.parents = [x]
    return z
end

function ^(x::Variable, y::Integer)
    x.numchildren += 1
    z = Variable(x.value ^ y)
    z.pjis = [y * x.value ^ (y - 1)]
    z.parents = [x]
    return z
end


convert(::Type{Variable}, x::Real) = Variable(x)

export Variable, backward!, grad, +, -, *, /, ^, convert

end


module FiniteDiff

function centralGrad(f, x, y; epsilon=0.0)

    if epsilon==0.0
        epsilon = eps(eltype(x)) ^ (1/3)
    end

    gradsx = (f(x .+ epsilon, y) - f(x .- epsilon, y)) / (2 * epsilon)
    gradsy = (f(x, y .+ epsilon) - f(x, y .- epsilon)) / (2 * epsilon)
    return gradsx, gradsy
end


function forwardGrad(f, x, y; epsilon=0.0)

    if epsilon==0.0
        epsilon = sqrt(eps(eltype(x)))
    end

    gradsx = (f(x .+ epsilon, y) - f(x, y)) / epsilon
    gradsy = (f(x, y .+ epsilon) - f(x, y)) / epsilon
    return gradsx, gradsy
end

export centralGrad, forwardGrad

end


module ComplexStep

function complexGrad(f, x, y; epsilon=0.0)

    if epsilon==0.0
        epsilon = eps(eltype(x))
    end

    xc = complex(x)
    yc = complex(y)
    gradsx = imag.(f(xc .+ epsilon * im, yc)) / epsilon
    gradsy = imag.(f(xc, yc .+ epsilon * im)) / epsilon
    return gradsx, gradsy
end

export complexGrad

end
