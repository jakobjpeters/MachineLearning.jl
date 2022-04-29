
function (layer::Layer)(h_params, input)
    layer.Zs = layer.weights * input
    if !isnothing(layer.biases)
        layer.Zs += layer.biases
    end

    return layer.Zs |> h_params.activ_func # |> layer.norm_func
end

function (neural_net::Neural_Network)(input, cache=false)
    for (layer, h_param) in zip(neural_net.layers, neural_net.h_params)
        input = layer(h_param, input)

        if layer === neural_net.layers[end] || cache
            layer.activations = input
        end
    end

    return neural_net.layers[end].activations
end

function backpropagate!(model, inputs, labels)
    for (input, label) in zip(inputs, labels)
        model(input, true)

        δl_δa = deriv(model.cost_func, model.layers[end].activations, label)
        prev_activations = input, map(layer -> layer.activations, model.layers[begin:end - 1])...
        activ_funcs = map(h_param -> h_param.activ_func, model.h_params)
        
        for (layer, activ_func, prev_activation) in zip(reverse(model.layers), reverse(activ_funcs), reverse(prev_activations))
            δl_δb = δl_δa .* deriv(activ_func, layer.Zs)

            layer.δl_δw -= δl_δb * transpose(prev_activation)
            if layer.biases !== nothing
                layer.δl_δb -= δl_δb
            end

            layer === model.layers[begin] && break

            δl_δa = transpose(layer.weights) * δl_δb
        end
    end

    return nothing
end

function apply_gradient!(layers, learn_rates, batch_size)
    scales = learn_rates / batch_size

    for (layer, scale) in zip(layers, scales)
        layer.weights += layer.δl_δw * scale 
        fill!(layer.δl_δw, 0.0)
        if layer.biases !== nothing
            layer.biases += layer.δl_δb * scale
            fill!(layer.δl_δb, 0.0)
        end
    end

    return nothing
end

function assess!(model, inputs, labels)
    correct = 0
    error = 0.0

    for (input, label) in zip(inputs, labels)
        output = model(input)

        if argmax(output) == label[1]
            correct += 1
        end

        error += mean(model.cost_func(output, label))
    end

    return correct / length(labels), error / length(labels)
end

function (epoch::Epoch)(model, inputs, labels)
    if epoch.shuffle && epoch.batch_size < length(inputs)
        inputs, labels = shuffle_data(inputs, labels)
    end

    for first in 1:epoch.batch_size:length(inputs)
        last = min(length(inputs), first + epoch.batch_size - 1)
        backpropagate!(model, view(inputs, first:last), view(labels, first:last))
        apply_gradient!(model.layers, map(h_params -> h_params.learn_rate, model.h_params), epoch.batch_size)
    end

    return nothing
end
