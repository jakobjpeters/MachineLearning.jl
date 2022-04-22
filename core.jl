
# too many allocations - try pre-allocation? try reassigning inputs and labels and return those
# type signature is iffy?
function shuffle_data(inputs, labels)
    data = collect(zip(inputs, labels))
    Random.shuffle!(data)
    return getindex.(data, 1), getindex.(data, 2)
end

function predict!(nn, input)
    nn.preallocs.as[begin] = input

    # use zip?
    for layer_n in 1:length(nn.layer_hparams.sizes)
        nn.preallocs.as[layer_n] = nn.layer_hparams.norm_funcs[layer_n](nn.preallocs.as[layer_n])

        nn.preallocs.zs[layer_n] = nn.params.ws[layer_n] * nn.preallocs.as[layer_n]
        if nn.layer_hparams.use_biases[layer_n]
            nn.preallocs.zs[layer_n] += nn.params.bs[layer_n]

        nn.preallocs.as[layer_n + 1] = nn.layer_hparams.activ_funcs[layer_n](nn.preallocs.zs[layer_n])
        end
    end
end

function assess!(nn, inputs, labels)
    correct = 0
    error = 0.0

    for (input, label) in zip(inputs, labels)
        predict!(nn, input)

        if issetequal(findall(x -> x > 0.5, nn.preallocs.as[end]), label)
            correct += 1
        end

        error += Statistics.mean(nn.model_hparams.cost_func(nn.preallocs.as[end], label))
    end

    return correct / length(labels), error / length(labels)
end

function backpropagate!(nn, inputs, labels)
    for (input, label) in zip(inputs, labels)
        predict!(nn, input)

        δl_δa = deriv(nn.model_hparams.cost_func, nn.preallocs.as[end], label)

        for layer_n in length(nn.layer_hparams.sizes):-1:1
            δl_δb = δl_δa .* deriv(nn.layer_hparams.activ_funcs[layer_n], nn.preallocs.zs[layer_n])

            # faster, but makes breaks the batching -> implement temp gradient allocation
            nn.params.ws[layer_n] -= nn.layer_hparams.learn_rates[layer_n] * δl_δb * transpose(nn.preallocs.as[layer_n])
            if nn.layer_hparams.use_biases[layer_n]
                nn.params.bs[layer_n] -= nn.layer_hparams.learn_rates[layer_n] * δl_δb
            end

            if layer_n != 1
                δl_δa = transpose(nn.params.ws[layer_n]) * δl_δb
            end

        end

    end
end

function print_assess(model, epoch, train_inputs, train_labels, test_inputs, test_labels)
    println("\nEpoch: ", epoch)
    # mse not type stable
    accuracy, mse = assess!(model, train_inputs, train_labels)
    println("    Train\tAccuracy: ", round(accuracy, digits = 4), "\t\tMSE: ", round(mse, digits = 8))
    # mse not type stable
    accuracy, mse = assess!(model, test_inputs, test_labels)
    println("    Test\tAccuracy: ", round(accuracy, digits = 4), "\t\tMSE: ", round(mse, digits = 8), "\n")
end

function train_epoch(nn, train_inputs, train_labels)
    if nn.model_hparams.shuffle && nn.model_hparams.batch_size != nn.model_hparams.input_size
        train_inputs, train_labels = shuffle_data(train_inputs, train_labels)
    end

    for j in 1:nn.model_hparams.batch_size:length(train_labels) - nn.model_hparams.batch_size
        backpropagate!(nn, train_inputs[j:j + nn.model_hparams.batch_size - 1], train_labels[j:j + nn.model_hparams.batch_size - 1])
    end
end
