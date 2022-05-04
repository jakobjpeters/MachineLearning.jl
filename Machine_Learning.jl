    
module Machine_Learning

# Testing


# External

# Files
import GZip
import ZipFile
import TOML

# Math
import Distributions: Normal
import Statistics: mean, std
import Random: shuffle!, seed!
import Printf: @printf

# GUI


# Internal
include("functions.jl")
include("emnist.jl")
include("utilities.jl")
include("types.jl")
include("core.jl")
include("interface.jl")

# parse 'config.TOML' and return useful data and datatypes
function load_config()
    config = TOML.parsefile("config.TOML")

    # extract dictionaries for readability
    seed = config["seed"]
    display = config["display"]
    data = config["data"]
    epochs = config["epochs"]
    model = config["model"]
    h_params = config["hyperparameters"]

    # helpers
    string_to_func = string -> getfield(@__MODULE__, Symbol(string))
    strings_to_funcs = strings -> map(string_to_func, strings)
    float = Dict("Float16" => Float16, "Float32" => Float32, "Float64" => Float64)
    h_params_args = zip(
        strings_to_funcs(h_params["normalization_functions"]),
        strings_to_funcs(h_params["activation_functions"]),
        h_params["learn_rates"]
    )

    # transform configuration 
    # see 'types.jl'
    sizes = model["sizes"][begin:end - 1]..., length(mapping(data["name"]))
    seed = seed == "missing" ? missing : seed
    display = string_to_func(display)
    dataset = load_dataset(data["name"], string_to_func(data["preprocessing_function"]), data["split_percentages"])
    epochs = map(i -> Epoch(epochs["batch_size"], parse(Bool, epochs["shuffle_data"]), string_to_func(epochs["cost_function"])), 1:epochs["iterations"])
    model = string_to_func(model["name"])(
        784, # input size, TODO: make dynamic
        float[model["precision"]],
        strings_to_funcs(model["weight_initialization_functions"]),
        sizes,
        model["use_biases"]
    )
    caches = make_caches(model)
    h_params = map(args -> Hyperparameters(args...), h_params_args)

    return config, seed, display, dataset, epochs, model, caches, h_params
end

function main()
    config, seed, display, dataset, epochs, model, caches, h_params = load_config()

    # seed is random if not specified
    ismissing(seed) || seed!(seed)

    # Unvectorize, since future code can't handle it yet
    # TODO: Vectorize future code
    inputs = [[col for col in eachcol(data.inputs)] for data in dataset]
    labels = [[col for col in eachcol(data.labels)] for data in dataset]
    dataset = [Data(input, label) for (input, label) in zip(inputs, labels)]

    # print configuration info
    display(config)
    # print pre-trained model assessment
    display(dataset, 0, model, epochs[begin].cost_func, h_params, caches)

    # main training loop
    # see 'core.jl' and 'interface.jl'
    for (i, epoch) in enumerate(epochs)
        # train model with data from first split
        @time epoch(model, h_params, caches, dataset[begin].inputs, dataset[begin].labels)
        @time display(dataset, i, model, epoch.cost_func, h_params, caches)
    end

    return config, model
end

main()

end