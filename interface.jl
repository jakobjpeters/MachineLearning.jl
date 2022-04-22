
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
