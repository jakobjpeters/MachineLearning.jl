
function assess!(model, inputs, labels)
    correct = 0
    error = 0.0

    for (input, label) in zip(inputs, labels)
        model(input)

        if argmax(model.layers[end].activations) == label[1]
            correct += 1
        end

        error += mean(model.cost_func(model.layers[end].activations, label))
    end

    return correct / length(labels), error / length(labels)
end

function backpropagate!(model, inputs, labels)
    for (input, label) in zip(inputs, labels)
        model(input)

        δl_δa = deriv(model.cost_func, model.layers[end].activations, label)

        for (i, layer) in enumerate(reverse(model.layers))
            δl_δb = δl_δa .* deriv(layer.activ_func, layer.Zs)

            prev_activations = layer === model.layers[begin] ? input : model.layers[end - i].activations

            layer.δl_δw -= δl_δb * transpose(prev_activations)
            if !isnothing(layer.biases)
                layer.δl_δb -= δl_δb
            end

            if layer === model.layers[begin]
                break
            end

            δl_δa = transpose(layer.weights) * δl_δb
        end
    end

    return nothing
end

function apply_gradient!(layers, batch_size)
    for layer in layers
        scale = layer.learn_rate / batch_size

        layer.weights += layer.δl_δw * scale 
        fill!(layer.δl_δw, 0.0)
        if !isnothing(layer.biases)
            layer.biases += layer.δl_δb * scale
            fill!(layer.δl_δb, 0.0)
        end
    end

    return nothing
end
