    
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

# GUI


# Internal
include("functions.jl")
include("types.jl")
include("core.jl")
include("emnist.jl")
include("print.jl")
include("interface.jl")

function main()
    config = TOML.parsefile("config.TOML")

    # use metaprogramming?
    seed = config["seed"]
    display = config["display"]
    data = config["data"]
    epochs = config["epochs"]
    model = config["model"]
    layers = config["layers"]

    string_to_func = string -> getfield(@__MODULE__, Symbol(string))
    strings_to_funcs = strings -> map(string_to_func, strings)
    float = Dict("Float16" => Float16, "Float32" => Float32, "Float64" => Float64)

    ismissing(seed) ? seed!() : seed!(seed)
    display = string_to_func(display)

    preprocess = string_to_func(data["preprocessing_function"])
    layer_sizes = vcat(layers["sizes"][begin:end - 1], length(mapping(data["name"])))

    dataset = load_dataset(data["name"], preprocess, data["split_percentages"])
    epochs = map(i -> Epoch(epochs["batch_size"], parse(Bool, epochs["shuffle_data"])), 1:epochs["n"])
    model = Neural_Network(
        string_to_func(model["cost_function"]),
        784, # make dynamic
        float[model["precision"]],
        strings_to_funcs(layers["weight_initialization_functions"]),
        strings_to_funcs(layers["normalization_functions"]),
        strings_to_funcs(layers["activation_functions"]),
        layers["learn_rates"],
        layer_sizes,
        layers["use_biases"]
    )

    display(data["name"], dataset, epochs, model)

end

main()

end