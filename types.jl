
struct Data{T, S, R<:Integer}
    inputs::T
    labels::S
    length::R
end

function Data(inputs, labels)
    if length(inputs) != length(labels)
        throw(ErrorException("inputs and labels must be the same length"))
    end

    Data(inputs, labels, length(inputs))
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

struct Epoch{T<:Integer}
    batch_size::T
    shuffle::Bool
end

function (epoch::Epoch)(model, inputs, labels)
    if epoch.shuffle && epoch.batch_size != model.input_size
        inputs, labels = shuffle_data(inputs, labels)
    end

    for i in 1:epoch.batch_size:length(inputs)
        backpropagate!(model, view(inputs, i:i + epoch.batch_size - 1), view(labels, i:i + epoch.batch_size - 1))
    end
end

mutable struct Layer
    weight_init_func
    norm_func
    activ_func
    weights
    biases
    δl_δw
    δl_δb
    Zs
    learn_rate
end

abstract type Model end

struct Neural_Network
    layers::Vector{Layer}
    cost_func
    input_size
    precision
    sizes
    activations
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
    Zs = [zeros(precision, size) for size in sizes]

    layers_args = zip(weight_init_funcs, norm_funcs, activ_funcs, weights, biases, δl_δw, δl_δb, Zs, learn_rates)
    layers = [Layer(layer_args...) for layer_args in layers_args]

    Neural_Network(
        layers,
        cost_func,
        input_size,
        precision,
        sizes,
        [zeros(precision, size) for size in tmp_sizes]
    )
end

function (neural_net::Neural_Network)(input)
    for (i, layer) in enumerate(neural_net.layers)
        prev_activations = layer === neural_net.layers[begin] ? input : neural_net.activations[i]
        layer.Zs = layer.weights * prev_activations
        if !isnothing(layer.biases)
            layer.Zs += layer.biases

        neural_net.activations[i + 1] = layer.Zs |> layer.activ_func # |> layer.norm_func
        end
    end
end

struct GAN <: Model
    generator
    discriminator
end
