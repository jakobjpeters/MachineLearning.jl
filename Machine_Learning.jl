    
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
include("types.jl")
include("core.jl")
include("emnist.jl")
include("interface.jl")


function unpack(seed, display, data, epochs, model, layers)
    string_to_func = string -> getfield(@__MODULE__, Symbol(string))
    strings_to_funcs = strings -> map(string_to_func, strings)
    float = Dict("Float16" => Float16, "Float32" => Float32, "Float64" => Float64)

    layers["sizes"][end] = length(mapping(data["name"]))
    display = string_to_func(display)

    dataset = load_dataset(data["name"], string_to_func(data["preprocessing_function"]), data["split_percentages"])
    epochs = map(i -> Epoch(epochs["batch_size"], parse(Bool, epochs["shuffle_data"])), 1:epochs["n"])
    model = Neural_Network(
        string_to_func(model["cost_function"]),
        784, # make dynamic
        float[model["precision"]],
        strings_to_funcs(layers["weight_initialization_functions"]),
        strings_to_funcs(layers["normalization_functions"]),
        strings_to_funcs(layers["activation_functions"]),
        layers["learn_rates"],
        layers["sizes"],
        layers["use_biases"]
    )

    return seed, display, dataset, epochs, model
end

function main()
    config = TOML.parsefile("config.TOML")
    seed, display, dataset, epochs, model = unpack(config["seed"], config["display"], config["data"], config["epochs"], config["model"], config["layers"])

    ismissing(seed) ? seed!() : seed!(seed)

    display(config)
    display(dataset, 0, model)

    # train neural net
    for (i, epoch) in enumerate(epochs)
        @time epoch(model, dataset[1].inputs, dataset[1].labels)
        display(dataset, i, model)
    end

    return config, model
end

main()

end