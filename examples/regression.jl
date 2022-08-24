
using MachineLearning

function main()
    x = convert.(Float32, collect(1:100))
    m, b = 3, -2
    y = map(input -> m * input + b, x)

    splits = [80, 20]
    datasets = split_dataset(Dataset(x, y), splits)

    loss = squared_error
    model = Linear(loss)

    # pre-trained
    @time terminal(assess(datasets, model))

    # see 'core.jl'
    @time train!(model, datasets[begin])

    # see 'interface.jl'
    @time terminal(assess(datasets, model))

    ρ = correlation_coefficient(x, y)
    println("Correlation coefficient: ", ρ)
    # TODO: pretty printing
    println(model)
end

main()
