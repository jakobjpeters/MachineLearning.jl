
function terminal(config, key="config")
    if config isa Dict
        for key in keys(config)
            terminal(config[key], key)
        end
    else
        println()
        println(key)
        println(config)
    end
end

# prints model assessment for each data split
function terminal(dataset, epoch, model, cost_func, h_params, caches)
    println("\nEpoch: ", epoch)
    # mse not type stable
    for (i, data) in enumerate(dataset)
        model(data.inputs, h_params, caches)
        accuracy, cost = assess(cost_func, last(caches).outputs, data.labels)
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
