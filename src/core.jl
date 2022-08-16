
using LinearAlgebra: BLAS.gemm!, axpy!

# calculate and cache linear and activation
function predict!(layer::Dense, x, activate, cache)
    cache.l = linear(layer.w, x, layer.b)
    
    cache.a = preallocate(cache.a, cache.l)
    map!(activate, cache.a, cache.l) # |> layer.norm_func

    return cache.a
end

# propagate input -> linear -> activation through each layer
function predict!(model::NeuralNetwork, x, layers_params, caches)
    for (layer, layer_params, cache) in zip(model.layers, layers_params, caches)
        x = predict!(layer, x, layer_params.activate, cache)
    end

    return caches[end].a
end

# regularize::typeof(T) where T <: Uniont{weight_decay, l1, l2} when 'l2' doesn't use an adaptive gradient
function update_weight!(regularize, λ, η, w, δe_δl, x)
    # gemm!(tA, tB, α, A, B, β, C) = α * A * B + β * C -> C where 'T' indicates a transpose
    gemm!('N', 'T', -η / size(x, 2), δe_δl, x, one(eltype(x)) - λ, w)
    
    return w
end

# slower, only call if 'λ != 0'
function update_weight!(regularize::typeof(l1), λ, η, w, δe_δl, x)
    ∇ = zeros(eltype(x))
    ∇ = δe_δl * transpose(x) / size(x, 2)
    penalty = derivative(regularize).(w, λ / η)
    # axpy!(α, X, Y) = α * X + Y -> Y
    axpy!(-η, ∇ + penalty, w)

    return w
end

# needed for 'l2' with adaptive gradient such as ADAM, although ADAM with weight decay is better
# https://arxiv.org/pdf/1711.05101v3.pdf
# function update_weight!(optimiser::T, regular_func::typeof(l2), λ, α, w, δe_δl, x)
#     ∇ = δe_δl * transpose(x) / size(x, 2)
#     penalty = derivative(regular_func)(w, λ / α)
#     # axpy!(α, X, Y) = α * X + Y -> Y
#     axpy!(-α, ∇ + penalty, w)
# end


@inline function update_params!(regularize, λ, η, w, δe_δl, x, b)
    update_weight!(regularize, λ, η, w, δe_δl, x)

    if !isnothing(b)
        δe_δb = dropdims(sum(δe_δl, dims = 2), dims = 2)
        axpy!(-η / size(x, 2), δe_δb, b)
    end

    return nothing
end

function train_layer!(i, x, layer, layer_params, cache, δe_δa)
    # calculate intermediate partial derivatives
    δa_δl = map(derivative(layer_params.activate), cache.l)

    cache.δe_δl = preallocate(cache.δe_δl, cache.l)
    cache.δe_δl .= δe_δa .* δa_δl

    # do not need to calculate δe_δa for first layer, since there are no parameters to update
    δe_δa = i == 1 ? nothing : transpose(layer.w) * cache.δe_δl

    update_params!(layer_params.regularize, layer_params.λ, layer_params.η, layer.w, cache.δe_δl, x, layer.b)

    # TODO: makes more sense to return 'layer'
    return δe_δa
end

# calculate the gradient and update the model's parameters for a batched input
@inline function backpropagate!(model, layers_params, caches, x, y, loss)
    predict!(model, x, layers_params, caches)

    δe_δa = derivative(loss)(y, caches[end].a)

    for i in reverse(eachindex(model.layers))
        layer_x = i == 1 ? x : caches[i - 1].a
        δe_δa = train_layer!(i, layer_x, model.layers[i], layers_params[i], caches[i], δe_δa)
    end

    return model
end

# test the model and return its accuracy and cost for each data split
function assess(dataset, model, loss, layers_params)
    accuracies = Vector{Float32}(undef, 0)
    costs = Vector{Float32}(undef, 0)

    # TODO: parameterize decision criterion
    criterion = pair -> argmax(pair[begin]) == argmax(pair[end])

    for data in dataset
        ŷ = model(data.x, layers_params)
        n = size(data.x, 2)

        n_correct = count(criterion, zip(eachcol(data.y), eachcol(ŷ)))
        push!(accuracies, n_correct / n)

        cost = sum(loss(data.y, ŷ))
        penalty = sum([sum(layers_params[i].regularize.(model.layers[i].w, layers_params[i].λ)) for i in eachindex(layers_params)])
        push!(costs, (cost + penalty) / n)
    end

    return @NamedTuple{accuracies::Vector{Float32}, costs::Vector{Float32}}((accuracies, costs))
end

# coordinate an epoch of model training
function train!(epoch, model, caches, x, y)
    n = size(x, 2)

    if epoch.shuffle && epoch.batch_size < n # TODO: && shuffle_pair!(x, y)
        x, y = shuffle_pair(x, y)
    end

    # train model for each batch
    for first in 1:epoch.batch_size:n
        last = min(n, first + epoch.batch_size - 1)
        prep_x = epoch.normalize(view(x, :, first:last))
        prep_y = view(y, :, first:last)

        backpropagate!(model, epoch.layers_params, caches, prep_x, prep_y, epoch.loss)
    end

    return model
end
