
struct Data{T, S<:Integer}
    inputs::T
    labels::T
    length::S
end

function split_dataset(inputs, labels, splits)
    if sum(splits) != 100
        throw(ErrorException("Splits must add to 100 (percent)"))
    end
    if length(inputs) != length(labels)
        throw(ErrorException("inputs and labels must be the same length"))
    end

    starts = Vector{Int64}()
    i = 0
    for split in splits
        if i < 100
            append!(starts, i)
        end
        i += split
    end
    stops = append!(starts[begin + 1:end], length(inputs))

    return [Data(view(inputs, start + 1:stop), view(labels, start + 1:stop), stop - start) for (start, stop) in zip(starts, stops)]
end


# struct Epoch
#     batch_size
#     shuffle
# end

# function (epoch::Epoch)(model, inputs, labels)
#     if epoch.shuffle && epoch.batch_size != model.input_size
#         inputs, labels = shuffle_data(train_inputs, train_labels)
#     end

#     model(inputs, labels)
# end

# struct Layer{T<:AbstractFloat, S<:Function, R<:Integer}
#     weight_init_func::S
#     # bias_init_func::S
#     norm_func::S
#     activ_func::S
#     size::R
#     use_biases::Bool
#     learn_rate::T
#     weights::Vector{Matrix{T}}
#     biases::Union{Nothing, Vector{Vector{T}}}
#     δl_δw::Vector{Matrix{T}}
#     δl_δb::Union{Nothing, Vector{Vector{T}}}
#     activations::Vector{Vector{T}}
#     zs::Vector{Vector{T}}
# end

# struct Neural_Network
#     layers::Vector{Layers}
#     cost_func
#     input_size
#     precision
# end

# function (neural_net::Neural_Network)(inputs) end


abstract type Model end
abstract type NeuralNetwork <: Model end
abstract type Hyperparameters end

struct LayerHyperparameters{T<:Function, S<:Function, R<:AbstractFloat, Q<:Function} <: Hyperparameters
    weight_init_funcs::Vector{Q}
    norm_funcs::Vector{T}
    activ_funcs::Vector{S}
    # fix? holy traits?
    learn_rates::Vector{R}
    sizes::Vector{Int64}
    use_biases::Vector{Bool}
end

function LayerHyperparameters(layer_hparams)
    # TODO: improve
    # TODO: include erroneous value
    if any(x -> x < 1, layer_hparams.sizes)
        throw(ErrorException("Invalid layer size. Must be greater than 0."))
    else
        LayerHyperparameters(
            layer_hparams.weight_init_funcs,
            layer_hparams.norm_funcs,
            layer_hparams.activ_funcs,
            layer_hparams.learn_rates,
            layer_hparams.sizes,
            layer_hparams.use_biases
        )
    end
end

struct ModelHyperparameters{T<:Function, S<:Integer} <: Hyperparameters
    cost_func::T
    epochs::S
    input_size::S
    batch_size::S
    precision::DataType # TODO: make precision a layer parameter?
    shuffle::Bool
    # optimizer::Function
    # regularizer::Function
end

function ModelHyperparameters(model_hparams)
    # TODO: improve
    # TODO: include erroneous value
    if !(model_hparams.precision <: AbstractFloat)
        throw(ErrorException("Invalid precision. Valid values are Float16, Float32, and Float64."))
    elseif model_hparams.epochs < 1
        throw(ErrorException("Invalid epochs. Must be greater than 0."))
    elseif model_hparams.batch_size < 1
        throw(ErrorException("Invalid batch size. Must be greater than 0."))
    else
        ModelHyperparameters(
            model_hparams.cost_func,
            model_hparams.epochs,
            model_hparams.input_size,
            model_hparams.batch_size,
            model_hparams.precision,
            model_hparams.shuffle,
        )
    end
end

struct Parameters{T<:AbstractFloat}
    ws::Vector{Matrix{T}}
    bs::Union{Nothing, Vector{Vector{T}}}
end

struct Preallocations{T<:AbstractFloat}
    δl_δw::Vector{Matrix{T}}
    δl_δb::Union{Nothing, Vector{Vector{T}}}
    as::Vector{Vector{T}}
    zs::Vector{Vector{T}}
end

# neural network
struct FFN <: NeuralNetwork
    model_hparams::ModelHyperparameters
    layer_hparams::LayerHyperparameters
    params::Parameters
    preallocs::Preallocations
end

# TODO: fix precision parameter - use holy traits?
# TODO: make sure layers match with hyperparams
function FFN(model_hparams, layer_hparams)
    sizes = model_hparams.input_size, layer_hparams.sizes...

    ws = [convert(Matrix{model_hparams.precision},
        rand(weight_init_func(input_size), output_size, input_size))
            for (weight_init_func, input_size, output_size) in zip(layer_hparams.weight_init_funcs, sizes[begin:end - 1], sizes[begin + 1:end])]
    bias_init = [use_bias ? zeros(model_hparams.precision, size) : nothing for (use_bias, size) in zip(layer_hparams.use_biases, layer_hparams.sizes)]

    FFN(
        model_hparams,
        layer_hparams,
        Parameters(
            ws,
            bias_init
        ),
        Preallocations(
            [convert(Matrix{model_hparams.precision},
                rand(weight_init_func(input_size), output_size, input_size))
                for (weight_init_func, input_size, output_size) in zip(layer_hparams.weight_init_funcs, sizes[begin:end - 1], sizes[begin + 1:end])],
            deepcopy(bias_init),
            [zeros(model_hparams.precision, size) for size in sizes],
            [zeros(model_hparams.precision, size) for size in sizes]
        )
    )
end

# TODO: parameterize ffn
struct GAN <: NeuralNetwork
    generator::FFN
    discriminator::FFN
end
