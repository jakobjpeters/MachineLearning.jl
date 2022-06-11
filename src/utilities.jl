
function percent_to_index(x, percent)
    return div(percent * size(x, 2), 100)
end

# given lists of inputs and labels, return a list of 'Data' split by percentages in 'splits'
function split_data(x, y, splits)
    sum(splits) != 100 && error("Splits must add to 100 (percent)")

    # generate indices to split the data by each percentage in 'splits'
    starts = Vector{Int64}(undef, 0)
    percent = 0

    for split in splits
        # append!(starts, div(percent * size(x, 2), 100))
        append!(starts, percent_to_index(x, percent))
        percent += split
    end

    stops = vcat(starts[begin + 1:end], size(x, 2))
    starts .+= 1

    return [Data(view(x, :, start:stop), view(y, :, start:stop)) for (start, stop) in zip(starts, stops)]
end

# load selected dataset, preprocess dataset, and return list of dataset splits
function load_dataset(name, preprocess, splits, precision)
    x, y = load_emnist(name)
    
    prep_x = mapslices(preprocess, convert.(precision, x), dims = 1)
    prep_y = convert.(precision, y)

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
    dataset_params = config["dataset_parameters"]
    epoch_params = config["epoch_parameters"]
    layers_params = config["layers_parameters"]
    model_params = config["model_parameters"]

    # helpers
    model_params["sizes"] = [model_params["sizes"][begin:end - 1]..., length(mapping(dataset_params["dataset"]))]
    string_to_func = string -> getfield(@__MODULE__, Symbol(string))
    strings_to_funcs = strings -> map(string_to_func, strings)
    float = Dict("Float32" => Float32, "Float64" => Float64)
    layers_parameters = zip(
        strings_to_funcs(layers_params["normalizers"]),
        strings_to_funcs(layers_params["activators"]),
        strings_to_funcs(layers_params["regularizers"]),
        convert.(float[config["precision"]], layers_params["regularize_rates"]),
        convert.(float[config["precision"]], layers_params["learn_rates"])
    )
    layers_parameters = map(layer_parameters -> LayerParameters(layer_parameters...), layers_parameters)
    n_epochs = epoch_params["number_of_epochs"]

    # seed is random if not specified
    # needs to be set before any random sampling
    seed == "missing" || seed!(seed)

    # transform configuration
    # see 'interface.jl'
    display = string_to_func(display)
    # see 'emnist.jl'
    dataset = load_dataset(dataset_params["dataset"], string_to_func(dataset_params["preprocessor"]), dataset_params["split_percentages"], float[config["precision"]])
    # see 'types.jl'
    epoch_params = EpochParameters(epoch_params["batch_size"], parse(Bool, epoch_params["shuffle_data"]), string_to_func(epoch_params["loss"]), string_to_func(epoch_params["normalizer"]), layers_parameters)
    model = string_to_func(model_params["model"])(
        # TODO: make input size dynamic
        784,
        float[config["precision"]],
        strings_to_funcs(model_params["weight_initializers"]),
        model_params["sizes"],
        model_params["use_biases"]
    )
    caches = map(_ -> Cache(float[config["precision"]]), eachindex(model.layers))

    return config, display, dataset, epoch_params, n_epochs, model, caches
end

# TODO: in-place?
function preallocate(x, dims)
    return size(x) == dims ? x : zeros(eltype(x), dims)
end
