
# Data

function split_data(inputs, labels, splits)
    sum(splits) != 100 && error("Splits must add to 100 (percent)")

    starts = Vector{Int64}()
    i = 0
    for split in splits
        if i < 100
            append!(starts, div(i * length(inputs), 100))
        end
        i += split
    end
    stops = append!(starts[begin + 1:end], length(inputs))
    starts .+= 1

    return [Data(view(inputs, start:stop), view(labels, start:stop)) for (start, stop) in zip(starts, stops)]
end

function load_dataset(name, preprocess, splits)
    dataset = load_emnist(name)
    prep_inputs = map(preprocess, dataset.inputs)
    return split_data(prep_inputs, dataset.labels, splits)
end

# make shuffle in place
function shuffle_data(inputs, labels)
    data = collect(zip(inputs, labels))
    shuffle!(data)
    return getindex.(data, 1), getindex.(data, 2)
end