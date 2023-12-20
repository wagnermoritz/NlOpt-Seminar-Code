using Flux, Images, MLDatasets, Plots, CUDA, ProgressMeter
using Flux: logitcrossentropy, onecold, onehotbatch, train!, withgradient
using LinearAlgebra, Random, Statistics, Printf


prec = Float64

x_train_raw, y_train_raw = MLDatasets.MNIST(Tx=prec, split=:train)[:]
x_test_raw, y_test_raw = MLDatasets.MNIST(Tx=prec, split=:test)[:]
x = zero(x_train_raw[:, :, 1])
x[1, 2] = 1
print(x)