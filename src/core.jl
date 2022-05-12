
# given input, calculate and cache 'Zs' and 'outputs'
function (dense::Dense)(inputs, activ_func, cache)
    cache.Zs = dense.weights * inputs
    if !isnothing(dense.biases)
        cache.Zs .+= dense.biases
    end

    cache.outputs = cache.Zs |> activ_func # |> layer.norm_func

    # TODO: should 'return nothing'
    return cache.outputs
end

# call each layer with its correct parameters
function (neural_net::Neural_Network)(inputs, h_params, caches)
    for (layer, h_param, cache) in zip(neural_net.layers, h_params, caches)
        # fix: allocating
        inputs = layer(inputs, h_param.activ_func, cache)
    end

    return nothing
end

# given a model, calculate and cache the gradient for a batch of inputs
@inline function backpropagate!(layers, cost_func, h_params, caches, inputs, labels)
    num_layers = length(layers)
    caches[num_layers].δl_δa = deriv(cost_func, labels, last(caches).outputs)
    
    # iterate end to begin to calculate each layer's gradient
    for i in reverse(1:num_layers)
        caches[i].δl_δz = caches[i].δl_δa .* deriv.(h_params[i].activ_func, caches[i].Zs)
        if i != 1
            caches[i - 1].δl_δa = transpose(layers[i].weights) * caches[i].δl_δz
        end

        # update weights and biases
        # dividing by batch size turns the gradients from a sum to an average
        scale = h_params[i].learn_rate / size(inputs, 2)
        # gemm!(tA, tB, alpha, A, B, beta, C) = alpha * A * B + beta * C -> C where 'T' transposes
        gemm!('N', 'T', scale, caches[i].δl_δz, i == 1 ? inputs : caches[i - 1].outputs, one(eltype(inputs)), layers[i].weights)
        if layers[i].biases !== nothing
            # axpy!(a, X, Y) = a * X + Y -> Y
            axpy!(scale, dropdims(sum(caches[i].δl_δz, dims = 2), dims = 2), layers[i].biases)
        end
    end

    return nothing
end

# given a model and data, test the model and return its accuracy and loss
function assess!(dataset, model, cost_func, h_params, caches)
    precision = eltype(first(model.layers).weights)
    accuracies = Vector{precision}()
    costs = Vector{precision}()

    for split in dataset
        model(split.inputs, h_params, caches)

        # TODO: parameterize decision criteria
        criteria = pair -> argmax(first(pair)) == argmax(last(pair))

        push!(accuracies, count(criteria, zip(eachcol(last(caches).outputs), eachcol(split.labels))) / size(last(caches).outputs, 2))
        push!(costs, mean(cost_func(split.labels, last(caches).outputs)))
    end

    return accuracies, costs
end

# given a model and data, coordinate model training
function (epoch::Epoch)(model, h_params, caches, inputs, labels)

    if epoch.shuffle && epoch.batch_size < size(inputs, 1)
        inputs, labels = shuffle_pair(inputs, labels)
    end

    # train model for each batch
    for first in 1:epoch.batch_size:size(inputs, 2)
        last = min(size(inputs, 2), first + epoch.batch_size - 1)

        model(view(inputs, :, first:last), h_params, caches)
        backpropagate!(model.layers, epoch.cost_func, h_params, caches, view(inputs, :, first:last), view(labels, :, first:last))
    end

    return nothing
end
