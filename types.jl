
struct Data
    inputs
    labels

    Data(inputs, labels) = length(inputs) == length(labels) ? new(inputs, labels) : error("Inputs and labels must be the same length")
end

struct Epoch{T<:Integer}
    batch_size::T
    shuffle::Bool
end

mutable struct Layer{T<:AbstractFloat}
    weights::Matrix{T}
    biases::Union{Vector{T}, Nothing}
    δl_δw::Matrix{T}
    δl_δb::Union{Vector{T}, Nothing}
    activations::Vector{T}
    Zs::Vector{T}
end

struct Hyperparameters{T<:Function, S<:Function, R<:AbstractFloat}
    norm_func::T
    activ_func::S
    learn_rate::R
end

abstract type Model end

struct Neural_Network{T<:Layer, S<:Hyperparameters, R<:Function} <: Model
    layers::Vector{T}
    h_params::Vector{S}
    cost_func::R
end

function Neural_Network(cost_func, input_size, precision, weight_init_funcs, norm_funcs, activ_funcs, learn_rates, sizes, use_biases)
    tmp_sizes = input_size, sizes...

    weights = [convert(Matrix{precision},
        rand(weight_init_func(input_size), output_size, input_size))
            for (weight_init_func, input_size, output_size) in zip(weight_init_funcs, tmp_sizes[begin:end - 1], tmp_sizes[begin + 1:end])]
    biases = [use_bias ? zeros(precision, size) : nothing for (use_bias, size) in zip(use_biases, sizes)]

    δl_δw = [convert(Matrix{precision},
        zeros(output_size, input_size))
            for (input_size, output_size) in zip(tmp_sizes[begin:end - 1], tmp_sizes[begin + 1:end])]
    δl_δb = deepcopy(biases)
    activations = [zeros(precision, size) for size in sizes]
    Zs = [zeros(precision, size) for size in sizes]

    layers_args = zip(weights, biases, δl_δw, δl_δb, activations, Zs)
    layers = [Layer(layer_args...) for layer_args in layers_args]

    h_params_args = zip(norm_funcs, activ_funcs, learn_rates)
    h_params = [Hyperparameters(h_param_args...) for h_param_args in h_params_args]

    Neural_Network(
        layers,
        h_params,
        cost_func
    )
end

struct GAN{T<:Model, S<:Model} <: Model
    generator::T
    discriminator::S
end
