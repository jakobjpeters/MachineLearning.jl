
function terminal(config)
    println()
    for key in keys(config)
        println(key, ":")
        println(config[key])
        println()
    end

    return nothing
end

# prints model assessment for each data split
function terminal(dataset, epoch, model, cost_func, h_params, caches)
    println("\nEpoch: ", epoch)
    # mse not type stable
    for (i, data) in enumerate(dataset)
        accuracy, cost = assess!(model, cost_func, h_params, caches, data.inputs, data.labels)
        @printf("\tSplit: %s\tAccuracy: %.4f\tCost: %.4f\n", i, accuracy, cost)
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
