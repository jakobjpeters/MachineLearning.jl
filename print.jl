
function print_data(data, dataset_name)
    print_images(data.inputs, data.labels, mapping(dataset_name), map(i -> rand(1:data.length), 1:10))
end

function print_info(dataset, epochs, model)
    println("Dataset: ", dataset)
    println()
    # println("Input size: ", model.input_size)
    println("Epochs: ", length(epochs))
    println("Batch size: ", map(epoch -> epoch.batch_size, epochs))
    println("Cost function: ", model.cost_func)
    println("Precision: ", model.precision)
    println("Shuffle: ", map(epoch -> epoch.shuffle, epochs))
    println()
    # println("Layer sizes: ", model.sizes)
    println("Learning rate: ", map(layer -> layer.learn_rate, model.layers))
    println("Use biases: ", map(layer -> !isnothing(layer.biases), model.layers))
    println("Activation functions: ", map(layer -> layer.activ_func, model.layers))
    println("Normalization functions: ", map(layer -> layer.norm_func, model.layers))
    # println("Weight Initialization functions: ", map(layer -> layer.weight_init_func, model.layers))
end

function print_assess(dataset, epoch, model)
    println("\nEpoch: ", epoch)
    # mse not type stable
    for (i, data) in enumerate(dataset)
        accuracy, loss = assess!(model, data.inputs, data.labels)
        println("    Split ", i, "\tAccuracy: ", round(accuracy, digits = 4), "\t\tLoss: ", round(loss, digits = 8))
    end
    println()
end