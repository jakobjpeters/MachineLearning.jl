
mutable struct Cache
    input::Matrix{Float32}
    linear::Matrix{Float32}

    Cache() = new()
end

struct Layer
    weights::Matrix{Float32}
    biases::Vector{Float32}

    Layer(input_size::Int, output_size::Int) = new(
        rand(Normal(Float32(0), input_size ^ -0.5), output_size, input_size),
        zeros(Float32, output_size)
    )
end

function (l::Layer)(activation, input::AbstractMatrix{Float32}, c::Cache)
    c.input, c.linear = input, muladd(l.weights, input, l.biases)
    activation.(c.linear)
end

function (l::Layer)(activation, input::AbstractMatrix{Float32})
    activation.(muladd(l.weights, input, l.biases))
end

struct NeuralNetwork{F}
    activation::F
    layers::Vector{Layer}

    NeuralNetwork(activation::F, sizes::Vector{Int}) where F = new{F}(
        activation,
        map(Layer, sizes, drop(sizes, 1))
    )
end

function (nn::NeuralNetwork)(input::AbstractMatrix{Float32}, caches::Vector{Cache})
    foldl(zip(nn.layers, caches); init = input) do layer_input, (layer, cache)
        layer(nn.activation, layer_input, cache)
    end
end

function (nn::NeuralNetwork)(input::AbstractMatrix{Float32})
    foldl((_input, layer) -> layer(nn.activation, _input), nn.layers; init = input)
end

function show(io::IO, ::MIME"text/plain", nn::NeuralNetwork)
    layers = nn.layers

    print(io, NeuralNetwork, '(', nn.activation, ", [", size(layers[begin].weights, 2), ", ")
    join(io, map(layer -> length(layer.biases), nn.layers), ", ")
    print(io, "])")
end
