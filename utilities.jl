
# given lists of inputs and labels, return a list of 'Data' split by percentages in 'splits'
function split_data(inputs, labels, splits)
    sum(splits) != 100 && error("Splits must add to 100 (percent)")

    # generate indices to split the data by each percentage in 'splits'
    starts = Vector{Int64}()
    i = 0
    for split in splits
        append!(starts, div(i * length(inputs), 100))
        i += split
    end
    stops = append!(starts[begin + 1:end], length(inputs))
    starts .+= 1

    return [Data(view(inputs, start:stop), view(labels, start:stop)) for (start, stop) in zip(starts, stops)]
end

# load selected dataset, preprocess dataset, and return list of dataset splits
function load_dataset(name, preprocess, splits)
    dataset = load_emnist(name)
    prep_inputs = map(preprocess, dataset.inputs)
    return split_data(prep_inputs, dataset.labels, splits)
end

# TODO: make shuffle in place
function shuffle_data(inputs, labels)
    data = collect(zip(inputs, labels))
    shuffle!(data)
    return getindex.(data, 1), getindex.(data, 2)
end