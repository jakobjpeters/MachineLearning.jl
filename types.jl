
struct Data{T1<:Number, T2<:Number}
    inputs::AbstractArray{T1}
    labels::AbstractArray{T2}
end

# functor, see 'core.jl'
struct Epoch{T<:Integer, F<:Function}
    batch_size::T
    shuffle::Bool
    cost_func::F
end

# corresponds to a layer in a 'Neural_Network'
mutable struct Cache{T<:AbstractFloat}
    δl_δw::Matrix{T}
    δl_δb::Union{Vector{T}, Nothing}
    activations::Matrix{T}
    Zs::Matrix{T}
end

# given a 'Neural_Network', construct a list of 'Cache's to prevent redundant calculations in 'core.jl'
function make_caches(neural_net)
    sizes = map(layer -> size(layer.weights, 1), neural_net.layers)

    δl_δw = map(layer -> fill!(deepcopy(layer.weights), 0.0), neural_net.layers)
    δl_δb = map(layer -> deepcopy(layer.biases), neural_net.layers)
    # can this be reduced due to changing batch sizes -> changing this size
    activations = map(size -> zeros(Float64, size, 10), sizes)
    Zs = deepcopy(activations)
    
    # TODO: remove splatting
    return map(args -> Cache(args...), zip(δl_δw, δl_δb, activations, Zs))
end

# corresponds to a layer in a 'Neural_Network'
struct Hyperparameters{F1<:Function, F2<:Function, T<:AbstractFloat}
    norm_func::F1
    activ_func::F2
    learn_rate::T
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

    # TODO: remove splatting
    layers = map(args -> Layer(args...), zip(weights, biases))
    return Neural_Network(layers)
end

# not implemented yet
struct GAN{T1<:Model, T2<:Model} <: Model
    generator::T1
    discriminator::T2
end
