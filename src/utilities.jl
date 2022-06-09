
# given lists of inputs and labels, return a list of 'Data' split by percentages in 'splits'
function split_data(input, label, splits)
    sum(splits) != 100 && error("Splits must add to 100 (percent)")

    # generate indices to split the data by each percentage in 'splits'
    starts = Vector{Int64}()
    i = 0
    for split in splits
        append!(starts, div(i * size(input, 2), 100))
        i += split
    end
    stops = append!(starts[begin + 1:end], size(input, 2))
    starts .+= 1

    return [Data(view(input, :, start:stop), view(label, :, start:stop)) for (start, stop) in zip(starts, stops)]
end

# load selected dataset, preprocess dataset, and return list of dataset splits
function load_dataset(name, preprocess, splits, precision)
    dataset = load_emnist(name)
    prep_input = mapslices(preprocess, convert.(precision, dataset.input), dims = 1)
    prep_label = convert.(precision, dataset.label)
    return split_data(prep_input, prep_label, splits)
end

# TODO: make this faster
function shuffle_pair(input, label)
    # pair inputs and labels, then shuffle the pairs
    data = shuffle!(collect(zip(eachcol(input), eachcol(label))))
    # unpack the pairs
    data = getindex.(data, 1), getindex.(data, 2)
    # transform to original structure
    return reduce(hcat, data[1]), reduce(hcat, data[2])
end

# parse 'config.TOML' and return useful data and datatypes
function load_config()
    config = TOML.parsefile(dirname(pwd()) * "/config.TOML")

    # extract dictionaries for readability
    seed = config["seed"]
    display = config["display"]
    data = config["data"]
    model = config["model"]
    epoch_param = config["epoch_parameter"]
    layer_param = config["layer_parameter"]

    # helpers
    config["sizes"] = model["sizes"][begin:end - 1]..., length(mapping(data["dataset"]))
    string_to_func = string -> getfield(@__MODULE__, Symbol(string))
    strings_to_funcs = strings -> map(string_to_func, strings)
    float = Dict("Float32" => Float32, "Float64" => Float64)
    layer_param_args = zip(
        strings_to_funcs(layer_param["normalization_functions"]),
        strings_to_funcs(layer_param["activation_functions"]),
        strings_to_funcs(layer_param["regularization_functions"]),
        convert.(float[config["precision"]], layer_param["regularization_rates"]),
        convert.(float[config["precision"]], layer_param["learn_rates"])
    )
    layer_param = map(layer_param_arg -> Layer_Parameter(layer_param_arg...), layer_param_args)
    num_epochs = epoch_param["number_of_epochs"]
    epoch_param = [epoch_param["batch_size"], parse(Bool, epoch_param["shuffle_data"]), string_to_func(epoch_param["cost_function"]), string_to_func(epoch_param["normalization_function"]), layer_param]

    # seed is random if not specified
    # needs to be set before any random sampling
    seed == "missing" || seed!(seed)

    # transform configuration 
    # see 'types.jl'
    display = string_to_func(display)
    dataset = load_dataset(data["dataset"], string_to_func(data["preprocessing_function"]), data["split_percentages"], float[config["precision"]])
    epoch_params = repeat([Epoch_Parameter(epoch_param...)], num_epochs)
    model = string_to_func(model["type"])(
        784, # input size, TODO: make dynamic
        float[config["precision"]],
        strings_to_funcs(model["weight_initialization_functions"]),
        config["sizes"],
        model["use_biases"]
    )
    caches = map(_ -> Cache(float[config["precision"]]), 1:length(model.layers))

    return config, display, dataset, epoch_params, model, caches
end

function preallocate(x, y)
    return size(x) == size(y) ? x : zeros(eltype(x), size(y))
end