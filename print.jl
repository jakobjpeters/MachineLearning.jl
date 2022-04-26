
function print_data(data, dataset_name)
    print_images(data.inputs, data.labels, mapping(dataset_name), [rand(1:data.length) for i in 1:10])
end

function print_info(dataset, ffn, epochs)
    println("Dataset: ", dataset)
    println()
    println("Input size: ", ffn.input_size)
    println("Epochs: ", length(epochs))
    println("Batch size: ", [epoch.batch_size for epoch in epochs])
    println("Cost function: ", ffn.cost_func)
    println("Precision: ", ffn.precision)
    println("Shuffle: ", [epoch.shuffle for epoch in epochs])
    println()
    println("Layer sizes: ", ffn.sizes)
    println("Learning rate: ", ffn.learn_rates)
    println("Use biases: ", ffn.use_biases)
    println("Activation functions: ", ffn.activ_funcs)
    println("Normalization functions: ", ffn.norm_funcs)
    println("Weight Initialization functions: ", ffn.weight_init_funcs)
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