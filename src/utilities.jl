
using TOML: parsefile
using Random: seed!, shuffle!

include("emnist.jl")

# given lists of inputs and labels, return a list of 'Data' split by percentages in 'splits'
function split_dataset(dataset::Dataset{A1, A2}, splits) where {A1, A2}
    split_i = map(split -> div(split * dataset.n, 100), splits[begin:end - 1])
    cumsum!(split_i, split_i)

    starts = prepend!(split_i .+ 1, 1)
    stops = append!(split_i, dataset.n)

    datasets = Dataset{A1, A2}[]
    for (start, stop) in zip(starts, stops)
        x_slice = dataset.x[repeat([:], length(size(dataset.x)) - 1)..., start:stop]
        y_slice = dataset.y[repeat([:], length(size(dataset.y)) - 1)..., start:stop]
        push!(datasets, Dataset(x_slice, y_slice))
    end

    return datasets
end

function split_dataset(dataset, splits)
    sum(splits) != 100 && error("Splits must add to 100 (percent)")
    return split_dataset(dataset, splits)
end

# load and preprocess selected dataset
function load_dataset(dataset_name, preprocess = identity)
    init_dataset(dataset_name)
    x, y = load_emnist(dataset_name)

    # mapslices is type-unstable
    prep_x = mapslices(preprocess, x, dims = 1)

    return Dataset(prep_x, y)
end

# TODO: make this faster
function shuffle_pair(x, y)
    # pair inputs and labels, then shuffle the pairs
    data = shuffle!(collect(zip(eachcol(x), eachcol(y))))
    # unpack the pairs
    data = getindex.(data, 1), getindex.(data, 2)
    # transform to original structure
    return reduce(hcat, data[begin]), reduce(hcat, data[end])
end

# reduces allocations if condition is met
# enables in-place operations for variables that may change size
function preallocate(x, y)
    return size(x) == size(y) ? x : similar(y)
end
