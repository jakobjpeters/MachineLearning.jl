
using TOML: parsefile
using Random: seed!, shuffle!

include("emnist.jl")

# given lists of inputs and labels, return a list of 'Data' split by percentages in 'splits'
function split_dataset(x, y, splits)
    sum(splits) != 100 && error("Splits must add to 100 (percent)")

    # generate indices to split the data by each percentage in 'splits'
    starts = Vector{Int64}(undef, 0)
    percent = 0

    for split in splits
        append!(starts, div(percent * size(x, 2), 100))
        percent += split
    end

    stops = vcat(starts[begin + 1:end], size(x, 2))
    starts .+= 1

    return [Dataset(view(x, :, start:stop), view(y, :, start:stop)) for (start, stop) in zip(starts, stops)]
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
