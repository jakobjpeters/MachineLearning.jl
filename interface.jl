
function terminal(dataset, model)
    # load data
    dir = pwd() * "/emnist/decompressed/"
    test_inputs = read_images(dir * dataset * "_test_images.bin", 16)
    test_labels = read_labels(dir * dataset * "_test_labels.bin", 8, 10)
    train_inputs = read_images(dir * dataset * "_train_images.bin", 16)
    train_labels = read_labels(dir * dataset * "_train_labels.bin", 8, 10)

    # print_inputs()
    print_info(dataset, model)

    print_assess(model, 0, train_inputs, train_labels, test_inputs, test_labels)

    # train neural net
    for epoch in 1:model.model_hparams.epochs
        @time train_epoch!(model, train_inputs, train_labels)
        @time print_assess(model, epoch, train_inputs, train_labels, test_inputs, test_labels)
    end
end

function gui(dataset, model_hparams, layer_hparams)
    throw(ErrorException("GUI not implemented yet."))
end
