
function print_data(data, dataset_name)
    print_images(data.inputs, data.labels, mapping(dataset_name), [rand(1:data.length) for i in 1:10])
end

function print_info(dataset, ffn)
    println("Dataset: ", dataset)
    println()
    println("Input size: ", ffn.model_hparams.input_size)
    println("Epochs: ", ffn.model_hparams.epochs)
    println("Batch size: ", ffn.model_hparams.batch_size)
    println("Cost function: ", ffn.model_hparams.cost_func)
    println("Precision: ", ffn.model_hparams.precision)
    println("Shuffle: ", ffn.model_hparams.shuffle)
    println()
    println("Layer sizes: ", ffn.layer_hparams.sizes)
    println("Learning rate: ", ffn.layer_hparams.learn_rates)
    println("Use biases: ", ffn.layer_hparams.use_biases)
    println("Activation functions: ", ffn.layer_hparams.activ_funcs)
    println("Normalization functions: ", ffn.layer_hparams.norm_funcs)
    println("Weight Initialization functions: ", ffn.layer_hparams.weight_init_funcs)
end

function print_assess(model, epoch, data_splits)
    println("\nEpoch: ", epoch)
    # mse not type stable
    for data in data_splits
        accuracy, loss = assess!(model, data.inputs, data.labels)
        println("    Split\tAccuracy: ", round(accuracy, digits = 4), "\t\tLoss: ", round(loss, digits = 8))
    end
    println()
end