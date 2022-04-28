
struct Data
    inputs
    labels

    Data(inputs, labels) = length(inputs) == length(labels) ? new(inputs, labels) : error("Inputs and labels must be the same length")
end

function split_data(inputs, labels, splits)
    if sum(splits) != 100
        throw(ErrorException("Splits must add to 100 (percent)"))
    end

    starts = Vector{Int64}()
    i = 0
    for split in splits
        if i < 100
            append!(starts, div(i * length(inputs), 100))
        end
        i += split
    end
    stops = append!(starts[begin + 1:end], length(inputs))

    return [Data(view(inputs, start + 1:stop), view(labels, start + 1:stop)) for (start, stop) in zip(starts, stops)]
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
        apply_gradient!(model.layers, epoch.batch_size)
    end

    return nothing
end

# mutable struct Layer{T<:Function, S<:Function, R<:Function, Q<:AbstractFloat, P::AbstractFloat}
mutable struct Layer{T<:Function, S<:Function, R<:AbstractFloat, Q<:AbstractFloat}
    norm_func::T
    activ_func::S
    weights::Matrix{R}
    biases::Vector{R}
    δl_δw::Matrix{R}
    δl_δb::Vector{R}
    activations::Vector{R}
    Zs::Vector{R}
    learn_rate::Q
end

abstract type Model end

struct Neural_Network
    layers::Vector{Layer}
    cost_func
    precision
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

    layers_args = zip(norm_funcs, activ_funcs, weights, biases, δl_δw, δl_δb, activations, Zs, learn_rates)
    layers = [Layer(layer_args...) for layer_args in layers_args]

    Neural_Network(
        layers,
        cost_func,
        precision,
    )
end

function (neural_net::Neural_Network)(input)
    for (i, layer) in enumerate(neural_net.layers)
        prev_activations = layer === neural_net.layers[begin] ? input : neural_net.layers[i - 1].activations
        layer.Zs = layer.weights * prev_activations
        if !isnothing(layer.biases)
            layer.Zs += layer.biases

        layer.activations = layer.Zs |> layer.activ_func # |> layer.norm_func
        end
    end

    return nothing
end

struct GAN <: Model
    generator
    discriminator
end
