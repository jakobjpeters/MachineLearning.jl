
# calculate and cache 'Z' and 'output'
function (dense::Dense)(input, activ_func, cache)
    cache.Z = dense.weight * input
    if !isnothing(dense.bias)
        cache.Z .+= dense.bias
    end

    cache.output = map(activ_func, cache.Z) # |> layer.norm_func
    # map!(activ_func, cache.output, cache.Z) # |> layer.norm_func

    return Nothing
end

# propagate input -> output through each layer
function (neural_net::Neural_Network)(input, layer_params, caches)
    for (layer, layer_param, cache) in zip(neural_net.layers, layer_params, caches)
        layer(input, layer_param.activ_func, cache)
        # TODO: remove this line
        input = cache.output
    end

    return nothing
end

# calculate the gradient and update the model's parameters for a batched input
@inline function backpropagate!(layers, cost_func, layer_params, caches, input, label)
    num_layers = length(layers)
    caches[num_layers].δl_δa = derivative(cost_func)(label, caches[end].output)

    for i in reverse(1:num_layers)
        # calculate intermediate partial derivatives
        δa_δz = map(derivative(layer_params[i].activ_func), caches[i].Z)
        caches[i].δl_δz = caches[i].δl_δa .* δa_δz
        # do not need to calculate δl_δa for first layer, since there are no parameters to update
        if i != 1
            caches[i - 1].δl_δa = transpose(layers[i].weight) * caches[i].δl_δz
        end

        layer_input = i == 1 ? input : caches[i - 1].output
        # negated to update with the negative gradient
        # dividing by batch size will average the gradients when updated
        scale = -layer_params[i].learn_rate / size(input, 2)

        # update weights and biases in-place
        # gemm!(tA, tB, α, A, B, β, C) = α * A * B + β * C -> C where 'T' indicates a transpose
        gemm!('N', 'T', scale, caches[i].δl_δz, layer_input, one(eltype(input)), layers[i].weight)
        if layers[i].bias !== nothing
            δl_δb = dropdims(sum(caches[i].δl_δz, dims = 2), dims = 2)
            # axpy!(α, X, Y) = α * X + Y -> Y
            axpy!(scale, δl_δb, layers[i].bias)
        end
    end

    return nothing
end

# test the model and return its accuracy and loss for each data split
function assess!(dataset, model, cost_func, layer_params, caches)
    precision = eltype(model.layers[begin].weight)
    accuracy = Vector{precision}()
    cost = Vector{precision}()

    for split in dataset
        model(split.input, layer_params, caches)

        # TODO: parameterize decision criteria
        criteria = pair -> argmax(pair[begin]) == argmax(pair[end])

        push!(accuracy, count(criteria, zip(eachcol(caches[end].output), eachcol(split.label))) / size(caches[end].output, 2))
        push!(cost, mean(cost_func(split.label, caches[end].output)))
    end

    return accuracy, cost
end

# coordinate an epoch of model training
function (epoch_param::Epoch_Parameter)(model, layer_params, caches, input, label)

    if epoch_param.shuffle && epoch_param.batch_size < size(input, 1)
        input, label = shuffle_pair(input, label)
    end

    # train model for each batch
    for first in 1:epoch_param.batch_size:size(input, 2)
        last = min(size(input, 2), first + epoch_param.batch_size - 1)
        norm_input = epoch_param.norm_func(view(input, :, first:last))

        model(norm_input, layer_params, caches)
        backpropagate!(model.layers, epoch_param.cost_func, layer_params, caches, norm_input, view(label, :, first:last))
    end

    return nothing
end
