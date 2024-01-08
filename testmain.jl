include("./testmodule.jl")
using .TestModule

x = Variable(1)
y = [x]
x.value = 2
print(x.value, y[1].value)