
const Assessment = @NamedTuple{accuracies, costs}

function indent(n)
    for i in 1:n
        print("\t")
    end
end

function terminal(config, n_indent = 0)
    if config isa Dict
        for key in keys(config)
            indent(n_indent)
            println(key)
            terminal(config[key], n_indent + 1)
        end
    else
        indent(n_indent)
        println(config)
        println()
    end

    return
end

# prints model assessment for each data split
function terminal(assessments::AbstractArray{T}) where T <: NamedTuple
    println("\nEpoch: ", length(assessments) - 1)
    for (i, assessment) in enumerate(zip(last(assessments).accuracies, last(assessments).costs))
        @printf("\tSplit: %s\tAccuracy: %.4f\tCost: %.4f\n", i, first(assessment), last(assessment))
    end
    println()

    return
end

function gui(config)
    return error("GUI not implemented yet.")
end

function gui(dataset, epoch, model)
    return error("GUI not implemented yet.")
end
