
function terminal(dataset_name, splits, model)
    
    dataset = load_dataset(dataset_name)
    data_splits = split_data(dataset, splits)

    # print_data(dataset, dataset_name)
    print_info(dataset_name, model)

    print_assess(model, 0, data_splits)

    # train neural net
    for epoch in 1:model.model_hparams.epochs
        @time train_epoch!(model, data_splits[1].inputs, data_splits[1].labels)
        @time print_assess(model, epoch, data_splits)
    end
end

function gui(datas, model_hparams, layer_hparams)
    throw(ErrorException("GUI not implemented yet."))
end
