
function terminal(config, key)
    println()
    println(key)
    println(config)
end

function terminal(config::Dict)
    for key in keys(config)
        terminal(config[key], key)
    end
end

# prints model assessment for each data split
function terminal(assessments)
    println("\nEpoch: ", length(assessments) - 1)
    for (i, assessment) in enumerate(zip(last(assessments).accuracies, last(assessments).costs))
        @printf("\tSplit: %s\tAccuracy: %.4f\tCost: %.4f\n", i, first(assessment), last(assessment))
    end
    println()

    return nothing
end

function gui(config)
    return error("GUI not implemented yet.")
end

function gui(dataset, epoch, model)
    return error("GUI not implemented yet.")
end
