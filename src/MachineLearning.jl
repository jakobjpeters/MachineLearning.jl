
module MachineLearning

import Base: show
using Distributions: Normal
using Downloads: download
using GZip: gzopen
using .Iterators: drop
using LinearAlgebra: axpy!, BLAS.gemm!, muladd
using Luxor: Drawing, Point, box, finish, setcolor
using Random: shuffle!
using ZipFile: Reader

include("neural_networks.jl")
include("math.jl")
include("emnist.jl")

function assess(loss, output, labels, n, epoch)
    _loss = round(sum(loss(output, labels)) / n; digits = 4)
    accuracy = round(count(splat(==), zip(identify(output), identify(labels))) / n; digits = 4)

    println("epoch\tloss\taccuracy\n$epoch\t$_loss\t$accuracy\n")
end

function train!(nn::NeuralNetwork, input::Matrix{Float32}, labels::Matrix{Float32};
    batch_size = 1,
    epochs::Integer = 10,
    learning_rate::AbstractFloat = 0.001,
    loss = squared_error
)
    activation_derivative, loss_derivative = derivative(nn.activation), derivative(loss)
    layers = reverse(nn.layers)
    caches = map(_ -> Cache(), layers)
    last_index = lastindex(layers)
    _learning_rate = -Float32(learning_rate)
    n = size(input, 2)

    assess(loss, nn(input), labels, n, 0)

    for epoch in 1:epochs
        for i in 1:batch_size:n
            is = i:min(n, i + batch_size - 1)
            @views batch_input, batch_labels = input[:, is], labels[:, is]
            δe_δa = loss_derivative(nn(batch_input, caches), batch_labels)

            for (i, (layer, cache)) in enumerate(zip(layers, caches))
                weights, biases = layer.weights, layer.biases
                adjusted_learning_rate = _learning_rate #/ Float32(size(cache.input, 2))
                δe_δl = δe_δa .* activation_derivative.(cache.linear)
                i == last_index || (δe_δa = weights' * δe_δl)

                gemm!('N', 'T', adjusted_learning_rate, δe_δl, cache.input, one, weights)
                axpy!(adjusted_learning_rate, sum(δe_δl, dims = 2), biases)
            end
        end

        assess(loss, nn(input), labels, n, epoch)
    end

    nn
end

export
    EMNIST, NeuralNetwork, balanced, by_class, by_merge, digits, letters, mnist,
    load_emnist, squared_error, relu, sigmoid, test_emnist, train_emnist, train!, z_score

end # MachineLearning
