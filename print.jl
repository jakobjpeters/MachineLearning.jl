
function print_data(data, dataset_name)
    print_images(data.inputs, data.labels, mapping(dataset_name), [rand(1:data.length) for i in 1:10])
end

function print_info(dataset, model, epochs)
    println("Dataset: ", dataset)
    println()
    println("Input size: ", model.input_size)
    println("Epochs: ", length(epochs))
    println("Batch size: ", [epoch.batch_size for epoch in epochs])
    println("Cost function: ", model.cost_func)
    println("Precision: ", model.precision)
    println("Shuffle: ", [epoch.shuffle for epoch in epochs])
    println()
    println("Layer sizes: ", model.sizes)
    println("Learning rate: ", model.learn_rates)
    println("Use biases: ", [!isnothing(layer.biases) for layer in model.layers])
    println("Activation functions: ", [layer.activ_func for layer in model.layers])
    println("Normalization functions: ", [layer.norm_func for layer in model.layers])
    println("Weight Initialization functions: ", [layer.weight_init_func for layer in model.layers])
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