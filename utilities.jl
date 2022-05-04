
# given lists of inputs and labels, return a list of 'Data' split by percentages in 'splits'
function split_data(inputs, labels, splits)
    sum(splits) != 100 && error("Splits must add to 100 (percent)")

    # generate indices to split the data by each percentage in 'splits'
    starts = Vector{Int64}()
    i = 0
    for split in splits
        append!(starts, div(i * size(inputs, 2), 100))
        i += split
    end
    stops = append!(starts[begin + 1:end], size(inputs, 2))
    starts .+= 1

    return [Data(view(inputs, :, start:stop), view(labels, :, start:stop)) for (start, stop) in zip(starts, stops)]
end

# load selected dataset, preprocess dataset, and return list of dataset splits
function load_dataset(name, preprocess, splits)
    dataset = load_emnist(name)
    prep_inputs = preprocess(dataset.inputs)
    return split_data(prep_inputs, dataset.labels, splits)
end

# TODO: make this faster
function shuffle_pair(inputs, labels)
    data = shuffle!(collect(zip(eachcol(inputs), labels)))
    data = getindex.(data, 1), getindex.(data, 2)
    return reduce(hcat, data[1]), reduce(hcat, data[2])
end