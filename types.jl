
struct Data
    inputs
    labels

    Data(inputs, labels) = length(inputs) == length(labels) ? new(inputs, labels) : error("Inputs and labels must be the same length")
end

function split_data(inputs, labels, splits)
    sum(splits) != 100 && error("Splits must add to 100 (percent)")

    starts = Vector{Int64}()
    i = 0
    for split in splits
        if i < 100
            append!(starts, div(i * length(inputs), 100))
        end
        i += split
    end
    stops = append!(starts[begin + 1:end], length(inputs))
    starts .+= 1

    return [Data(view(inputs, start:stop), view(labels, start:stop)) for (start, stop) in zip(starts, stops)]
end

function load_dataset(name, preprocess, splits)
    dataset = load_emnist(name)
    prep_inputs = map(preprocess, dataset.inputs)
    return split_data(prep_inputs, dataset.labels, splits)
end

# make shuffle in place
function shuffle_data(inputs, labels)
    data = collect(zip(inputs, labels))
    shuffle!(data)
    return getindex.(data, 1), getindex.(data, 2)
end

struct Epoch{T<:Integer}
    batch_size::T
    shuffle::Bool
end

function (epoch::Epoch)(model, inputs, labels)
    if epoch.shuffle && epoch.batch_size < length(inputs)
        inputs, labels = shuffle_data(inputs, labels)
    end

    for first in 1:epoch.batch_size:length(inputs)
        last = min(length(inputs), first + epoch.batch_size - 1)
        backpropagate!(model, view(inputs, first:last), view(labels, first:last))
        apply_gradient!(model.layers, map(h_params -> h_params.learn_rate, model.h_params), epoch.batch_size)
    end

    return nothing
end

mutable struct Layer{T<:AbstractFloat}
    weights::Matrix{T}
    biases::Union{Vector{T}, Nothing}
    δl_δw::Matrix{T}
    δl_δb::Union{Vector{T}, Nothing}
    activations::Vector{T}
    Zs::Vector{T}
end

function (layer::Layer)(h_params, input)
    layer.Zs = layer.weights * input
    if !isnothing(layer.biases)
        layer.Zs += layer.biases
    end

    return layer.Zs |> h_params.activ_func # |> layer.norm_func
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

function (neural_net::Neural_Network)(input, cache=false)
    for (layer, h_param) in zip(neural_net.layers, neural_net.h_params)
        input = layer(h_param, input)

        if layer === neural_net.layers[end] || cache
            layer.activations = input
        end
    end

    return neural_net.layers[end].activations
end

struct GAN{T<:Model, S<:Model} <: Model
    generator::T
    discriminator::S
end
