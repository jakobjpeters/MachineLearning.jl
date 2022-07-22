
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
function load_dataset(dataset_name, preprocess, precision)
    init_dataset(dataset_name)
    x, y = load_dataset(dataset_name)

    # mapslices is type-unstable
    prep_x = mapslices(preprocess, convert.(precision, x), dims = 1)
    prep_y = convert.(precision, y)

    return Dataset(prep_x, prep_y)
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

# TODO: in-place?
function preallocate(x, dims)
    return size(x) == dims ? x : zeros(eltype(x), dims)
end
