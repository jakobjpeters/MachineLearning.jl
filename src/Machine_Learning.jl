    
module Machine_Learning

# Testing


# External

# Files
import GZip
import ZipFile
import TOML

# Math
import Distributions: Normal
import Statistics: stdm
import Random: shuffle!, seed!
import Printf: @printf
import LinearAlgebra: BLAS.gemm!, axpy!

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
    config = TOML.parsefile(dirname(pwd()) * "/config.TOML")

    # extract dictionaries for readability
    seed = config["seed"]
    display = config["display"]
    data = config["data"]
    epochs = config["epochs"]
    model = config["model"]
    h_params = config["hyperparameters"]

    # helpers
    config["sizes"] = model["sizes"][begin:end - 1]..., length(mapping(data["dataset"]))
    string_to_func = string -> getfield(@__MODULE__, Symbol(string))
    strings_to_funcs = strings -> map(string_to_func, strings)
    float = Dict("Float32" => Float32, "Float64" => Float64)
    h_params_args = zip(
        strings_to_funcs(h_params["normalization_functions"]),
        strings_to_funcs(h_params["activation_functions"]),
        convert.(float[config["precision"]], h_params["learn_rates"])
    )
    # transform configuration 
    # see 'types.jl'
    # seed is random if not specified
    seed = seed == "missing" ? missing : seed
    # needs to be set before other data strcutures are initialized
    ismissing(seed) || seed!(seed)
    display = string_to_func(display)
    dataset = load_dataset(data["dataset"], string_to_func(data["preprocessing_function"]), data["split_percentages"], float[config["precision"]])
    epochs = repeat([Epoch(epochs["batch_size"], parse(Bool, epochs["shuffle_data"]), string_to_func(epochs["cost_function"]))], epochs["num_epochs"])
    model = string_to_func(model["type"])(
        784, # input size, TODO: make dynamic
        float[config["precision"]],
        strings_to_funcs(model["weight_initialization_functions"]),
        config["sizes"],
        model["use_biases"]
    )
    caches = map(_ -> Cache(float[config["precision"]]), 1:length(model.layers))
    h_params = map(args -> Hyperparameters(args...), h_params_args)

    return config, display, dataset, epochs, model, caches, h_params
end

function main()
    config, display, dataset, epochs, model, caches, h_params = load_config()

    assessment = @NamedTuple{accuracies, costs}
    assessments = [assessment(assess!(dataset, model, first(epochs).cost_func, h_params, caches))]

    display(config)
    display(assessments)

    # main training loop
    # see 'core.jl' and 'interface.jl'
    @time for epoch in epochs
        # train model with data from first split
        @time epoch(model, h_params, caches, dataset[begin].inputs, dataset[begin].labels)
        @time push!(assessments, assessment(assess!(dataset, model, first(epochs).cost_func, h_params, caches)))
        display(assessments)
    end

    return config, model
end

main()

end