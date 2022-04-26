
function terminal(dataset_name, splits, model, epochs)
    
    dataset = load_dataset(dataset_name)
    data_splits = split_data(dataset, splits)

    # print_data(dataset, dataset_name)
    print_info(dataset_name, model, epochs)

    print_assess(model, 0, data_splits)

    # train neural net
    for (i, epoch) in enumerate(epochs)
        @time epoch(model, data_splits[1].inputs, data_splits[1].labels)
        @time print_assess(model, i, data_splits)
    end
end

function gui(datas, model_hparams, layer_hparams)
    throw(ErrorException("GUI not implemented yet."))
end
