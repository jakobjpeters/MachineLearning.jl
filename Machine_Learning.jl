    
module Machine_Learning

# Testing
using InteractiveUtils
using BenchmarkTools

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


function load_config()
    config = TOML.parsefile("config.TOML")
    seed = config["seed"]
    display = config["display"]
    data = config["data"]
    epochs = config["epochs"]
    model = config["model"]
    layers = config["layers"]

    string_to_func = string -> getfield(@__MODULE__, Symbol(string))
    strings_to_funcs = strings -> map(string_to_func, strings)
    float = Dict("Float16" => Float16, "Float32" => Float32, "Float64" => Float64)

    layers["sizes"][end] = length(mapping(data["name"]))
    seed = seed == "missing" ? missing : seed
    display = string_to_func(display)

    cost_func = string_to_func(model["cost_function"])
    dataset = load_dataset(data["name"], string_to_func(data["preprocessing_function"]), data["split_percentages"])
    epochs = map(i -> Epoch(epochs["batch_size"], parse(Bool, epochs["shuffle_data"])), 1:epochs["n"])
    model = Neural_Network(
        784, # make dynamic
        float[model["precision"]],
        strings_to_funcs(layers["weight_initialization_functions"]),
        layers["sizes"],
        layers["use_biases"]
    )
    caches = get_caches(model)
    h_params_args = zip(
        strings_to_funcs(layers["normalization_functions"]),
        strings_to_funcs(layers["activation_functions"]),
        layers["learn_rates"]
    )
    h_params = map(args -> Hyperparameters(args...), h_params_args)

    return config, seed, display, dataset, epochs, model, cost_func, caches, h_params
end

function main()
    config, seed, display, dataset, epochs, model, cost_func, caches, h_params = load_config()

    ismissing(seed) || seed!(seed)

    display(config)
    display(dataset, 0, model, cost_func, h_params, caches)

    for (i, epoch) in enumerate(epochs)
        @time epoch(model, cost_func, h_params, caches, dataset[1].inputs, dataset[1].labels)
        display(dataset, i, model, cost_func, h_params, caches)
    end

    return config, model
end

main()

end