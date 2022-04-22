
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

function terminal(dataset, model_hparams, layer_hparams)
    # load data
    dir = pwd() * "/emnist/decompressed/"
    test_inputs = read_images(dir * dataset * "_test_images.bin", 16)
    test_labels = read_labels(dir * dataset * "_test_labels.bin", 8, 10)
    train_inputs = read_images(dir * dataset * "_train_images.bin", 16)
    train_labels = read_labels(dir * dataset * "_train_labels.bin", 8, 10)

    ffn = FFN(model_hparams, layer_hparams)

    # print_inputs()
    print_info(dataset, ffn)

    print_assess(ffn, 0, train_inputs, train_labels, test_inputs, test_labels)

    # train neural net
    for epoch in 1:ffn.model_hparams.epochs
        @time train_epoch(ffn, train_inputs, train_labels)
        @time print_assess(ffn, epoch, train_inputs, train_labels, test_inputs, test_labels)
    end
end

function gui(dataset, model_hparams, layer_hparams)
    throw(ErrorException("GUI not implemented yet."))
end
