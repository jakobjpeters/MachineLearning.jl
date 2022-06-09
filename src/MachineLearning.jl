    
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

function main()
    config, display, dataset, epoch_params, model, caches = load_config()

    assessment = @NamedTuple{accuracies, costs}
    assessments = [assessment(assess!(dataset, model, first(epoch_params).cost_func, first(epoch_params).layer_param, caches))]

    display(config)
    # pre-trained
    display(assessments)

    # main training loop
    # see 'core.jl' and 'interface.jl'
    @time for epoch_param in epoch_params
        @time epoch_param(model, epoch_param.layer_param, caches, dataset[begin].input, dataset[begin].label)
        @time push!(assessments, assessment(assess!(dataset, model, epoch_param.cost_func, epoch_param.layer_param, caches)))
        display(assessments)
    end

    return config, model
end

main()

end
