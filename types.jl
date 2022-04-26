
struct Data{T, S<:Integer}
    inputs::T
    labels::T
    length::S
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

function split_data(data, splits)
    return split_data(data.inputs, data.labels, splits)
end

struct Epoch # {T<:Integer}
    batch_size # ::T
    shuffle::Bool

    Epoch(; batch_size = 1, shuffle = true) = new(batch_size, shuffle)
end

function (epoch::Epoch)(model, inputs, labels)
    if epoch.shuffle && epoch.batch_size != model.input_size
        inputs, labels = shuffle_data(inputs, labels)
    end

    for i in 1:epoch.batch_size:length(inputs)
        backpropagate!(model, view(inputs, i:i + epoch.batch_size - 1), view(labels, i:i + epoch.batch_size - 1))
    end
end

struct Layer{T<:AbstractFloat, S<:Function, R<:Integer}
    weight_init_func::S
    # bias_init_func::S
    norm_func::S
    activ_func::S
    input_size::R
    output_size::R
    use_biases::Bool
    learn_rate::T
    weights::Vector{Matrix{T}}
    δl_δw::Vector{Matrix{T}}
    biases::Union{Nothing, Vector{Vector{T}}}
    δl_δb::Union{Nothing, Vector{Vector{T}}}
    activations::Vector{Vector{T}}
    Zs::Vector{Vector{T}}

    Layer(; kwargs...) = new{T, S, R}(
        kwargs.weight_init_func,
        kwargs.norm_func,
        kwargs.activ_func,
        kwargs.input_size,
        kwargs.output_size,
        kwargs.use_biases,
        kwargs.learn_rate,
        rand(weight_init_func(input_size), output_size, input_size), # weights
        zeros(output_size, input_size), # δl_δw
        kwargs.use_bias ? zeros(model_hparams.precision, output_size) : nothing, # biases
        kwargs.use_bias ? zeros(model_hparams.precision, output_size) : nothing, # δl_δb
        zeros(output_size), # activations
        zeros(output_size) # Zs
    )
end

abstract type Model end

struct Neural_Network{T<:Function, S<:AbstractFloat, R<:Integer} <: Model
    cost_func::T
    precision::S
    input_size::R
    layers::Vector{Layer}
end

function (neural_net::Neural_Network)(inputs) end

struct FFN
    cost_func
    input_size
    precision
    weight_init_funcs
    norm_funcs
    activ_funcs
    learn_rates
    sizes
    use_biases::Vector{Bool}
    weights
    biases
    δl_δw
    δl_δb
    activations
    Zs
end

# TODO: fix precision parameter - use holy traits?
# TODO: make sure layers match with hyperparams
function FFN(cost_func, input_size, precision, weight_init_funcs, norm_funcs, activ_funcs, learn_rates, sizes, use_biases)
    tmp_sizes = input_size, sizes...

    weights = [convert(Matrix{precision},
        rand(weight_init_func(input_size), output_size, input_size))
            for (weight_init_func, input_size, output_size) in zip(weight_init_funcs, tmp_sizes[begin:end - 1], tmp_sizes[begin + 1:end])]
    biases = [use_bias ? zeros(precision, size) : nothing for (use_bias, size) in zip(use_biases, sizes)]

    FFN(
        cost_func,
        input_size,
        precision,
        weight_init_funcs,
        norm_funcs,
        activ_funcs,
        learn_rates,
        sizes,
        use_biases,
        weights,
        biases,
        [convert(Matrix{precision},
            rand(weight_init_func(input_size), output_size, input_size))
            for (weight_init_func, input_size, output_size) in zip(weight_init_funcs, tmp_sizes[begin:end - 1], tmp_sizes[begin + 1:end])],
        deepcopy(biases),
        [zeros(precision, size) for size in tmp_sizes],
        [zeros(precision, size) for size in tmp_sizes]
    )
end

struct GAN <: Model
    generator
    discriminator
end
