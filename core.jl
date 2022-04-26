
function assess!(model, inputs, labels)
    correct = 0
    error = 0.0

    for (input, label) in zip(inputs, labels)
        model(input)

        if issetequal(findall(x -> x > 0.5, model.activations[end]), label)
            correct += 1
        end

        error += mean(model.cost_func(model.activations[end], label))
    end

    return correct / length(labels), error / length(labels)
end

function backpropagate!(model, inputs, labels)
    for (input, label) in zip(inputs, labels)
        model(input)

        δl_δa = deriv(model.cost_func, model.activations[end], label)

        for layer_n in reverse(1:length(model.sizes))
            δl_δb = δl_δa .* deriv(model.activ_funcs[layer_n], model.Zs[layer_n])

            model.δl_δw[layer_n] -= δl_δb * transpose(model.activations[layer_n])
            if model.use_biases[layer_n]
                model.δl_δb[layer_n] -= δl_δb
            end

            if layer_n != 1
                δl_δa = transpose(model.weights[layer_n]) * δl_δb
            end
        end
    end

    for layer_n in 1:length(model.sizes)
        model.weights[layer_n] += model.learn_rates[layer_n] * model.δl_δw[layer_n]
        fill!(model.δl_δw[layer_n], 0.0)

        if model.use_biases[layer_n]
            model.biases[layer_n] += model.learn_rates[layer_n] * model.δl_δb[layer_n]
            fill!(model.δl_δb[layer_n], 0.0)
        end
    end
end
