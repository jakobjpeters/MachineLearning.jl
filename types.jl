
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

# neural network
struct FFN{T<:AbstractFloat} <: NeuralNetwork
    model_hparams::ModelHyperparameters
    layer_hparams::LayerHyperparameters
    weighted_input::Vector{Vector{T}}
    activations::Vector{Vector{T}}
    weights::Vector{Matrix{T}}
    biases::Union{Nothing, Vector{Vector{T}}}
end

# TODO: fix precision parameter - use holy traits?
# TODO: make sure layers match with hyperparams
function FFN(model_hparams, layer_hparams)
    sizes = model_hparams.input_size, layer_hparams.sizes...

    FFN(
        model_hparams,
        layer_hparams,
        [zeros(model_hparams.precision, size) for size in sizes],
        [zeros(model_hparams.precision, size) for size in sizes],
        # TODO: make readable
        [convert(Matrix{model_hparams.precision},
            rand(weight_init_func(input_size), output_size, input_size))
                for (weight_init_func, input_size, output_size) in zip(layer_hparams.weight_init_funcs, sizes[begin:end - 1], sizes[begin + 1:end])],
        [use_bias ? zeros(model_hparams.precision, size) : nothing for (use_bias, size) in zip(layer_hparams.use_biases, layer_hparams.sizes)]
    )
end

# TODO: parameterize ffn
struct GAN <: NeuralNetwork
    generator::FFN
    discriminator::FFN
end
