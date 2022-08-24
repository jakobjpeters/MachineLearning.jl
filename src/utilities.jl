
using TOML: parsefile
using Random: seed!, shuffle!

include("emnist.jl")

function init_caches(n)
    return map(_ -> Cache(), 1:n)
end

function init_layers_params(n, η, regularizer = Regularizer(), normalize = identity)
    params = []
    for param in [η, normalize, regularizer]
        new_param = param
        if isa(param, AbstractArray)
            if length(param) != n
                throw(ErrorException("Parameters must be a single value or have length(param) == n"))
            end
        else
            new_param = repeat([param], n)
        end
        push!(params, new_param)
    end

    return map(param -> LayerParameters(param...), zip(params...))
end

# given lists of inputs and labels, return a list of 'Data' split by percentages in 'splits'
function split_dataset(dataset::Dataset{A1, A2}, splits) where {A1, A2}
    sum(splits) != 100 && error("Splits must add to 100 (percent)")

    split_i = map(split -> div(split * dataset.n, 100), splits[begin:end - 1])
    cumsum!(split_i, split_i)

    starts = prepend!(split_i .+ 1, 1)
    stops = append!(split_i, dataset.n)

    datasets = Dataset{A1, A2}[]
    for (start, stop) in zip(starts, stops)
        x = dataset.x[repeat([:], length(size(dataset.x)) - 1)..., start:stop]
        y = dataset.y[repeat([:], length(size(dataset.y)) - 1)..., start:stop]
        push!(datasets, Dataset(x, y))
    end

    return datasets
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
function shuffle(dataset)
    # pair inputs and labels, then shuffle the pairs
    xy = shuffle!(collect(zip(eachcol(dataset.x), eachcol(dataset.y))))
    # unpack the pairs
    x, y = getindex.(xy, 1), getindex.(xy, 2)
    # transform to original structure
    return Dataset(reduce(hcat, x), reduce(hcat, y))
end

# reduces allocations if condition is met
# enables in-place operations for variables that may change size
function preallocate(x, y)
    return size(x) == size(y) ? x : similar(y)
end