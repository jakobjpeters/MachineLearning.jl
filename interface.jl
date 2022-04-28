
function terminal(dataset_name, dataset, epochs, model)

    print_info(dataset_name, epochs, model)

    print_assess(dataset, 0, model)

    # train neural net
    for (i, epoch) in enumerate(epochs)
        @time epoch(model, dataset[1].inputs, dataset[1].labels)
        @time print_assess(dataset, i, model)
    end
end

function gui(dataset, epochs, model)
    throw(ErrorException("GUI not implemented yet."))
end
