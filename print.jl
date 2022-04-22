
function print_assess(model, epoch, train_inputs, train_labels, test_inputs, test_labels)
    println("\nEpoch: ", epoch)
    # mse not type stable
    accuracy, mse = assess!(model, train_inputs, train_labels)
    println("    Train\tAccuracy: ", round(accuracy, digits = 4), "\t\tMSE: ", round(mse, digits = 8))
    accuracy, mse = assess!(model, test_inputs, test_labels)
    println("    Test\tAccuracy: ", round(accuracy, digits = 4), "\t\tMSE: ", round(mse, digits = 8), "\n")
end

function print_inputs()
    print_images(test_inputs, test_labels, map, [rand(1:length(test_labels)) for i in 1:10])
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