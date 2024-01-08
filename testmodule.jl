module TestModule

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

# overload operators to build up reverse AD graph
import Base: +, -, *, /, ^, convert, promote_type

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

# here we assume that there are never non-default fields in variables that still have to be promoted
# function convert(::Type{Variable{T}}, x::Variable{S}) where {T<:Real, S<:Real}
#     return Variable(convert(T, x.value))
# end

convert(::Type{Variable{T}}, x::Real) where {T<:Real} = Variable(convert(T, x))
promote_type(::Variable{T}, ::Variable{S}) where {T<:Real, S<:Real} = Variable{promote_type(T, S)}

export Variable, +, -, *, /, ^, convert, promote_type

end