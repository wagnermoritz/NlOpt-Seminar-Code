module ReverseAD

# struct holding a value and all information needed for reverse AD
mutable struct Variable{T} <: Number
    value::T
    adjoint::T
    pjis::Vector{T}
    children::Vector{Variable{T}}

    function Variable(value)
        x = new{typeof(value)}()
        x.value = value
        x.adjoint = zero(value)
        x.pjis = typeof(value)[]
        x.children = typeof(x)[]
        return x
    end
end

# set all adjoints except the adjoint of x to zero
function zero_adjoints!(x::Variable)
    for i in 1:length(x.children)
        x.children[i].adjoint = zero(x.children[i].adjoint)
        zero_adjoints!(x.children[i])
    end
    return
end

# accumulate derivatives in all adjoints
function recursive_backward!(x::Variable)
    for i in 1:length(x.children)
        x.children[i].adjoint += x.pjis[i] * x.adjoint
        recursive_backward!(x.children[i])
    end
    return
end

# reverse sweep
function backward!(x::Variable)
    x.adjoint = one(x.adjoint)
    zero_adjoints!(x)
    recursive_backward!(x)
end

function grad(x::Variable)
    x.adjoint
end

# overload operators to build up reverse AD graph
import Base: +, -, *, /, ^, convert

function +(x::Variable, y::Variable)
    z = Variable(x.value + y.value)
    z.pjis = [one(x.value), one(y.value)]
    z.children = [x, y]
    return z
end

function +(x::Variable, y::Real)
    z = Variable(x.value + y)
    z.pjis = [one(x.value)]
    z.children = [x]
    return z
end

function +(x::Real, y::Variable)
    return y + x
end

function -(x::Variable, y::Variable)
    z = Variable(x.value - y.value)
    z.pjis = [one(x.value), -one(y.value)]
    z.children = [x, y]
    return z
end

function -(x::Variable, y::Real)
    z = Variable(x.value - y)
    z.pjis = [one(x.value)]
    z.children = [x]
    return z
end

function -(x::Real, y::Variable)
    return -y + x
end

function -(x::Variable)
    z = Variable(-x.value)
    z.pjis = [-one(x.value)]
    z.children = [x]
    return z
end

function *(x::Variable, y::Variable)
    z = Variable(x.value * y.value)
    z.pjis = [y.value, x.value]
    z.children = [x, y]
    return z
end

function *(x::Variable, y::Real)
    z = Variable(x.value * y)
    z.pjis = [y]
    z.children = [x]
    return z
end

function *(x::Real, y::Variable)
    return y * x
end

function /(x::Variable, y::Variable)
    z = Variable(x.value / y.value)
    z.pjis = [1 / y.value, -x.value / (y.value ^ 2)]
    z.children = [x, y]
    return z
end

function /(x::Variable, y::Real)
    z = Variable(x.value / y)
    z.pjis = [1 / y]
    z.children = [x]
    return z
end

function /(x::Real, y::Variable)
    z = Variable(x / y.value)
    z.pjis = [-x / (y.value ^ 2)]
    z.children = [y]
    return z
end

function ^(x::Variable, y::AbstractFloat)
    z = Variable(x.value ^ y)
    z.pjis = [y * x.value ^ (y - 1)]
    z.children = [x]
    return z
end

function ^(x::Variable, y::Integer)
    z = Variable(x.value ^ y)
    z.pjis = [y * x.value ^ (y - 1)]
    z.children = [x]
    return z
end


convert(::Type{Variable}, x::Real) = Variable(x)

export Variable, backward!, grad, +, -, *, /, ^, convert

end


module CentralFD

function gradFD(f, x, y)
    epsilon = eps(eltype(x)) ^ (1/3)
    gradsx = zero(x)
    gradsy = zero(y)
    gradsx .= (f(x .+ epsilon, y) - f(x .- epsilon, y)) / (2 * epsilon)
    gradsy .= (f(x, y .+ epsilon) - f(x, y .- epsilon)) / (2 * epsilon)
    return (gradsx, gradsy)
end

export gradFD

end
