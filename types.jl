
struct Data
    inputs
    labels

    Data(inputs, labels) = length(inputs) == length(labels) ? new(inputs, labels) : error("Inputs and labels must be the same length")
end

# functor, see 'core.jl'
struct Epoch{T<:Integer, S<:Function}
    batch_size::T
    shuffle::Bool
    cost_func::S
end

# corresponds to a layer in a 'Neural_Network'
mutable struct Cache{T<:AbstractFloat}
    δl_δw::Matrix{T}
    δl_δb::Union{Vector{T}, Nothing}
    activations::Vector{T}
    Zs::Vector{T}
end

# given a 'Neural_Network', construct a list of 'Cache's to prevent redundant calculations in 'core.jl'
function make_caches(neural_net)
    sizes = map(layer -> size(layer.weights, 1), neural_net.layers)

    δl_δw = map(layer -> fill!(deepcopy(layer.weights), 0.0), neural_net.layers)
    δl_δb = map(layer -> deepcopy(layer.biases), neural_net.layers)
    activations = map(size -> zeros(Float64, size), sizes)
    Zs = deepcopy(activations)
    
    return map(args -> Cache(args...), zip(δl_δw, δl_δb, activations, Zs))
end

# corresponds to a layer in a 'Neural_Network'
struct Hyperparameters{T<:Function, S<:Function, R<:AbstractFloat}
    norm_func::T
    activ_func::S
    learn_rate::R
end

# functor, see 'core.jl'
mutable struct Layer{T<:AbstractFloat}
    weights::Matrix{T}
    biases::Union{Vector{T}, Nothing}
end

abstract type Model end

# functor, see 'core.jl'
struct Neural_Network{T<:Layer} <: Model
    layers::Vector{T}
end

function Neural_Network(input_size, precision, weight_init_funcs, sizes, use_biases)
    tmp_sizes = input_size, sizes...

    # TODO: improve readability
    weights = [convert(Matrix{precision},
        rand(weight_init_func(input_size), output_size, input_size))
            for (weight_init_func, input_size, output_size) in zip(weight_init_funcs, tmp_sizes[begin:end - 1], tmp_sizes[begin + 1:end])]
    biases = [use_bias ? zeros(precision, size) : nothing for (use_bias, size) in zip(use_biases, sizes)]

    layers = map(args -> Layer(args...), zip(weights, biases))
    return Neural_Network(layers)
end

# not implemented yet
struct GAN{T<:Model, S<:Model} <: Model
    generator::T
    discriminator::S
end
