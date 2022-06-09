
# given lists of inputs and labels, return a list of 'Data' split by percentages in 'splits'
function split_data(x, y, splits)
    sum(splits) != 100 && error("Splits must add to 100 (percent)")

    # generate indices to split the data by each percentage in 'splits'
    starts = Vector{Int64}()
    i = 0

    for split in splits
        append!(starts, div(i * size(x, 2), 100))
        i += split
    end

    stops = append!(starts[begin + 1:end], size(x, 2))
    starts .+= 1

    return [Data(view(x, :, start:stop), view(y, :, start:stop)) for (start, stop) in zip(starts, stops)]
end

# load selected dataset, preprocess dataset, and return list of dataset splits
function load_dataset(name, preprocess, splits, precision)
    dataset = load_emnist(name)
    
    prep_x = mapslices(preprocess, convert.(precision, dataset.x), dims = 1)
    prep_y = convert.(precision, dataset.y)

    return split_data(prep_x, prep_y, splits)
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

# parse 'config.TOML' and return useful data and datatypes
# TODO: improve readability
function load_config()
    config = TOML.parsefile(dirname(pwd()) * "/config.TOML")

    # extract dictionaries for readability
    seed = config["seed"]
    display = config["display"]
    data = config["data"]
    model = config["model"]
    epoch = config["epoch"]
    layers_parameters = config["layers_parameters"]

    # helpers
    model["sizes"] = [model["sizes"][begin:end - 1]..., length(mapping(data["dataset"]))]
    string_to_func = string -> getfield(@__MODULE__, Symbol(string))
    strings_to_funcs = strings -> map(string_to_func, strings)
    float = Dict("Float32" => Float32, "Float64" => Float64)
    layers_parameters = zip(
        strings_to_funcs(layers_parameters["normalizers"]),
        strings_to_funcs(layers_parameters["activators"]),
        strings_to_funcs(layers_parameters["regularizers"]),
        convert.(float[config["precision"]], layers_parameters["regularize_rates"]),
        convert.(float[config["precision"]], layers_parameters["learn_rates"])
    )
    layers_parameters = map(layer_parameters -> LayerParameters(layer_parameters...), layers_parameters)
    n_epochs = epoch["number_of_epochs"]
    epoch = [epoch["batch_size"], parse(Bool, epoch["shuffle_data"]), string_to_func(epoch["loss"]), string_to_func(epoch["normalizer"]), layers_parameters]

    # seed is random if not specified
    # needs to be set before any random sampling
    seed == "missing" || seed!(seed)

    # transform configuration
    # see 'interface.jl'
    display = string_to_func(display)
    # see 'emnist.jl'
    dataset = load_dataset(data["dataset"], string_to_func(data["preprocessor"]), data["split_percentages"], float[config["precision"]])
    # see 'types.jl'
    epoch = Epoch(epoch...)
    model = string_to_func(model["model"])(
        # TODO: make input size dynamic
        784,
        float[config["precision"]],
        strings_to_funcs(model["weight_initializers"]),
        model["sizes"],
        model["use_biases"]
    )
    caches = map(_ -> Cache(float[config["precision"]]), eachindex(model.layers))

    return config, display, dataset, epoch, n_epochs, model, caches
end

function preallocate(x, y)
    return size(x) == size(y) ? x : zeros(eltype(x), size(y))
end