using Flux, Images, MLDatasets, Plots, CUDA, ProgressMeter
using Flux: logitcrossentropy, onecold, onehotbatch, train!, withgradient
using LinearAlgebra, Random, Statistics, Printf

#Random.seed!(1)

x_train_raw, y_train_raw = MLDatasets.MNIST(Tx=Float32, split=:train)[:]
x_test_raw, y_test_raw = MLDatasets.MNIST(Tx=Float32, split=:test)[:]

#colorview(Gray, x_train_raw[:, :, 1]')

x_train = Flux.flatten(x_train_raw)
y_train = onehotbatch(y_train_raw, 0:9)
trainloader = Flux.DataLoader((x_train, y_train), batchsize=128, shuffle=false)

x_test = Flux.flatten(x_test_raw)
y_test = onehotbatch(y_test_raw, 0:9)
testloader = Flux.DataLoader((x_test, y_test), batchsize=128, shuffle=false)

# define model architecture
# no softmax since we use logitcrossentropy loss for numerical stability
model = Chain(
    Dense(28 * 28, 256, σ),
    #Dense(256, 256, σ),
    Dense(256, 128, σ),
    #Dense(128, 128, σ),
    Dense(128, 64, σ),
    Dense(64, 10)
)


function lossfn(model, x, y)
    return logitcrossentropy(model(x), y)
end


function get_gradient_AD(lossfn, model, x, y)
    return withgradient(() -> lossfn(model, x, y), Flux.params(model))
end


function get_gradients_FD(lossfn, model, x, y)
    epsilon = sqrt(eps(eltype(x)))
    val = lossfn(model, x, y)
    grads = Dict()
    for (j, p) in enumerate(Flux.params(model))
        s = size(p)
        p = reshape(p, :)
        grads[p] = zero(p)
        for i = 1:size(p, 1)
            p_tmp = p[i]
            p[i] += epsilon
            Flux.params(model)[j] = reshape(p, s) #fuck
            val_i = lossfn(model, x, y)
            grads[p][i] = (val_i - val) ./ epsilon
            p[i] = p_tmp
            Flux.params(model)[j] = reshape(p, s)
        end
        grads[p] = reshape(grads[p], s)
    end
    return val, grads
end


function testmodel(model, testloader, numbatches=10)
    acc = 0
    n = 0
    for (i, (x, y)) in enumerate(testloader)
        if i == numbatches + 1
            break
        end
        pred = onecold(model(x))
        acc += sum(pred .== onecold(y))
        n += size(x, 2)
    end
    return acc / n
end


function SGD!(model, trainloader, testloader, lossfn; γ=0.1, λ=0.001, μ=0.2, τ=0.1, epochs=50)
    """
    Stochastic gradient descent wrt. the parameters of the model.

    γ: step size
    λ: weight decay
    μ: momentum
    τ: dampening
    """
    #params = Flux.params(model)
    losses = Float32[]
    test_accs = Float32[]
    train_accs = Float32[]
    t = 1

    for epoch in 1:epochs
        print("\rEpoch $epoch/$epochs")
        epoch_loss = 0
        n = 0
        for (x, y) in trainloader
            loss, grads = get_gradients_FD(lossfn, model, x, y)
            params = Flux.params(model)
            if t == 1
                global b = copy(grads) #there has to be another way
            end
            t += 1
            epoch_loss += loss
            n += size(x, 2)

            for p in params
                if λ != 0.0
                    grads[p] .+= λ .* p
                end
                if μ != 0.0
                    b[p] = μ .* b[p] .+ (1 - τ) .* grads[p]
                    grads[p] .= b[p]
                end
                p .-= γ .* grads[p]
            end
        end
        push!(losses, epoch_loss / n)
        push!(test_accs, testmodel(model, testloader))
        push!(train_accs, testmodel(model, trainloader))
    end
    return losses, test_accs, train_accs
end


epochs = 50
losses, test_accs, train_accs = SGD!(model, trainloader, testloader, lossfn, epochs=epochs)

p1 = plot(range(1, epochs, length=epochs), losses, ylabel="loss")
p2 = plot(range(1, epochs, length=epochs), test_accs, xlabel="epochs", ylabel="test acc.", name="test accuracy")
p3 = plot(range(1, epochs, length=epochs), train_accs, xlabel="epochs", ylabel="train acc.", name="train accuracy")
plot(p1, p2, p3, layout=(3, 1), legend=false)