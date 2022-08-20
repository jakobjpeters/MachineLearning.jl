
using LinearAlgebra: BLAS.gemm!, axpy!

# calculate and cache linear and activation
function predict!(layer::Dense, x, activate, cache)
    cache.l = linear(layer.w, x, layer.b)
    
    cache.a = preallocate(cache.a, cache.l)
    map!(activate, cache.a, cache.l) # |> layer.norm_func

    return cache.a
end

# propagate input -> linear -> activation through each layer
function predict!(model::NeuralNetwork, x, activators, caches)
    for (layer, activate, cache) in zip(model.layers, activators, caches)
        x = predict!(layer, x, activate, cache)
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
# function update_weight!(regularize::typeof(l1), λ, η, w, δe_δl, x)
#     ∇ = zeros(eltype(x))
#     ∇ = δe_δl * transpose(x) / size(x, 2)
#     penalty = derivative(regularize)(w, λ / η)
#     # axpy!(α, X, Y) = α * X + Y -> Y
#     axpy!(-η, ∇ + penalty, w)

#     return w
# end

# needed for 'l2' with adaptive gradient such as ADAM, although ADAM with weight decay is better
# https://arxiv.org/pdf/1711.05101v3.pdf
# function update_weight!(optimiser::T, regular_func::typeof(l2), λ, α, w, δe_δl, x)
#     ∇ = δe_δl * transpose(x) / size(x, 2)
#     penalty = derivative(regular_func)(w, λ / α)
#     # axpy!(α, X, Y) = α * X + Y -> Y
#     axpy!(-α, ∇ + penalty, w)
# end


@inline function update_params!(regularizer, η, w, δe_δl, x, b)
    update_weight!(regularizer.regularize, regularizer.λ, η, w, δe_δl, x)

    if !isnothing(b)
        δe_δb = dropdims(sum(δe_δl, dims = 2), dims = 2)
        axpy!(-η / size(x, 2), δe_δb, b)
    end

    return nothing
end

# calculate the gradient and update the model's parameters for a batched input
@inline function backpropagate!(model, layers_params, caches, x, y, loss)
    predict!(model, x, layers_params.activators, caches)

    δe_δa = derivative(loss)(y, caches[end].a)

    for i in reverse(eachindex(model.layers))
        layer_x = i == 1 ? x : caches[i - 1].a
        
        # calculate intermediate partial derivatives
        δa_δl = map(derivative(layers_params.activators[i]), caches[i].l)

        caches[i].δe_δl = preallocate(caches[i].δe_δl, caches[i].l)
        caches[i].δe_δl .= δe_δa .* δa_δl

        # do not need to calculate δe_δa for first layer, since there are no parameters to update
        δe_δa = i == 1 ? nothing : transpose(model.layers[i].w) * caches[i].δe_δl

        update_params!(layers_params.regularizers[i], layers_params.η[i], model.layers[i].w, caches[i].δe_δl, layer_x, model.layers[i].b)
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
        ŷ = model(data.x, layers_params.activators)
        n = size(data.x, 2)

        n_correct = count(criterion, zip(eachcol(data.y), eachcol(ŷ)))
        push!(accuracies, n_correct / n)

        cost = sum(loss(data.y, ŷ))
        penalty = sum([sum(regularizer.regularize(layer.w, regularizer.λ)) for (layer, regularizer) in zip(model.layers, layers_params.regularizers)])
        push!(costs, (cost + penalty) / n)
    end

    return @NamedTuple{accuracies::Vector{Float32}, costs::Vector{Float32}}((accuracies, costs))
end

function assess(dataset, model, loss)
    costs = Vector{Float32}(undef, 0)

    for data in dataset
        ŷ = model(data.x)
        n = size(data.x, 2)

        cost = sum(loss(data.y, ŷ))
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

function train!(model::Linear{<:AbstractVector}, x, y)
    throw(ErrorException("Multiple regression not implemented yet"))

    return model
end

function train!(model::Linear, dataset)
    x̄ = mean(dataset.x)
    ȳ = mean(dataset.y)

    model.w = sum((dataset.x .- x̄) .* (dataset.y .- ȳ)) / sum((dataset.x .- x̄) .^ 2)
    if !isnothing(model.b)
        model.b = ȳ - x̄ * model.w
    end

    return model
end