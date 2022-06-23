    
module Machine_Learning

import InteractiveUtils: @which, @code_warntype

include("math.jl")
include("types.jl")
include("utilities.jl")
include("interface.jl")
include("core.jl")

function main()
    config, display, dataset, epoch, n_epochs, model, caches = load_config()

    display(config)

    # pre-trained
    assessments = [assess(dataset, model, epoch.loss, epoch.layers_params)]
    display(assessments)

    # main training loop
    # see 'core.jl'
    @time for i in 1:n_epochs
        @time train!(epoch, model, caches, dataset[begin].x, dataset[begin].y)

        @time assessment = assess(dataset, model, epoch.loss, epoch.layers_params)
        push!(assessments, assessment)
        # see 'interface.jl'
        display(assessments)
    end

    return
end

main()

end
