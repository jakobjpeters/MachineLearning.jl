
function print_data(data, dataset_name)
    print_images(data.inputs, data.labels, mapping(dataset_name), [rand(1:data.length) for i in 1:10])
end

function print_info(dataset, neural_net, epochs)
    println("Dataset: ", dataset)
    println()
    println("Input size: ", neural_net.input_size)
    println("Epochs: ", length(epochs))
    println("Batch size: ", [epoch.batch_size for epoch in epochs])
    println("Cost function: ", neural_net.cost_func)
    println("Precision: ", neural_net.precision)
    println("Shuffle: ", [epoch.shuffle for epoch in epochs])
    println()
    println("Layer sizes: ", neural_net.sizes)
    println("Learning rate: ", neural_net.learn_rates)
    println("Use biases: ", [!isnothing(biases) for biases in neural_net.biases])
    println("Activation functions: ", [layer.activ_func for layer in neural_net.layers])
    println("Normalization functions: ", [layer.norm_func for layer in neural_net.layers])
    println("Weight Initialization functions: ", [layer.weight_init_func for layer in neural_net.layers])
end

function print_assess(model, epoch, data_splits)
    println("\nEpoch: ", epoch)
    # mse not type stable
    for (i, data) in enumerate(data_splits)
        accuracy, loss = assess!(model, data.inputs, data.labels)
        println("    Split ", i, "\tAccuracy: ", round(accuracy, digits = 4), "\t\tLoss: ", round(loss, digits = 8))
    end
    println()
end