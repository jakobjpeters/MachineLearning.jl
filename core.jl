
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

        for i in reverse(1:length(model.layers))
            δl_δb = δl_δa .* deriv(model.layers[i].activ_func, model.Zs[i])

            model.δl_δw[i] -= δl_δb * transpose(model.activations[i])
            if !isnothing(model.biases)
                model.δl_δb[i] -= δl_δb
            end

            if i == 1
                break
            end

            δl_δa = transpose(model.weights[i]) * δl_δb
        end
    end

    for i in 1:length(model.layers)
        model.weights[i] += model.learn_rates[i] * model.δl_δw[i]
        fill!(model.δl_δw[i], 0.0)

        if !isnothing(model.biases)
            model.biases[i] += model.learn_rates[i] * model.δl_δb[i]
            fill!(model.δl_δb[i], 0.0)
        end
    end
end
