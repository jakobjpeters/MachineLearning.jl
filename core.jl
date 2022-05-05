
# 'Layer' functor
# given input, calculate and cache 'Zs' and 'activations'
function (dense::Dense)(input, activ_func, cache)
    cache.Zs = dense.weights * input
    if !isnothing(dense.biases)
        cache.Zs .+= dense.biases
    end

    cache.activations = cache.Zs |> activ_func # |> layer.norm_func

    # should 'return nothing'
    return cache.activations
end

# 'Neural_Network' functor
# call each layer with its correct parameters
function (neural_net::Neural_Network)(input, h_params, caches)
    for (layer, h_param, cache) in zip(neural_net.layers, h_params, caches)
        # fix: allocating
        input = layer(input, h_param.activ_func, cache)
    end

    return nothing
end

# given a model, calculate and cache the gradient for a batch of inputs
function backpropagate(layers, cost_func, h_params, caches, inputs, labels)
    # fix: allocating
    δl_δa = deriv(cost_func, caches[end].activations, labels)
    prev_activations = pushfirst!(map(cache -> cache.activations, caches[begin:end - 1]), inputs)
    
    # iterate end to begin to calculate each layer's gradient
    for (layer, cache, prev_activation, h_param) in zip(reverse(layers), reverse(caches), reverse(prev_activations), reverse(h_params))
        δl_δz = δl_δa .* deriv(h_param.activ_func, cache.Zs)

        # cache gradients
        # gradients are averaged (divided by 'batch_size') in 'apply_gradient!' for efficiency
        mul!(cache.δl_δw, δl_δz, transpose(prev_activation))
        if layer.biases !== nothing
            cache.δl_δb = dropdims(sum(δl_δz, dims = 2), dims = 2)
        end

        # if first layer, this calculation is not needed
        layer === layers[begin] && break
        δl_δa = transpose(layer.weights) * δl_δz
    end

    return nothing
end

# update weights and biases with cached gradient
function apply_gradient!(layers, learn_rates, caches, batch_size)
    # dividing by 'batch_size' turns the gradients from a sum to an average
    scales = learn_rates / batch_size

    # update each layer's weights and biases, then reset its cache
    for (layer, cache, scale) in zip(layers, caches, scales)
        # in-place a * X + Y, stored in Y
        axpy!(scale, cache.δl_δw, layer.weights)
        if layer.biases !== nothing
            axpy!(scale, cache.δl_δb, layer.biases)
        end
    end

    return nothing
end

# given a model and data, test the model and return its accuracy and loss
function assess(cost_func, inputs, output, labels)
    # TODO: parameterize decision criteria
    criteria = pair -> argmax(first(pair)) == argmax(last(pair))
    accuracy = count(criteria, zip(eachcol(output), eachcol(labels))) / size(inputs, 2)

    loss = pair -> mean(cost_func(first(pair), last(pair)))
    cost = mean(map(loss, zip(eachcol(output), eachcol(labels))))

    return accuracy, cost
end

# 'Epoch' functor
# given a model and data, coordinate model training
function (epoch::Epoch)(model, h_params, caches, inputs, labels)

    if epoch.shuffle && epoch.batch_size < size(inputs, 1)
        inputs, labels = shuffle_pair(inputs, labels)
    end

    # train model for each batch
    for first in 1:epoch.batch_size:size(inputs, 2)
        last = min(size(inputs, 2), first + epoch.batch_size - 1)
        model(view(inputs, :, first:last), h_params, caches)
        backpropagate(model.layers, epoch.cost_func, h_params, caches, view(inputs, :, first:last), view(labels, :, first:last))
        
        apply_gradient!(model.layers, map(h_params -> h_params.learn_rate, h_params), caches, epoch.batch_size)
    end

    return nothing
end
