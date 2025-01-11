
@enum EMNIST balanced by_class by_merge digits letters mnist

const cache = Dict{EMNIST, Pair{NTuple{4, Matrix{Float32}}, Int}}()
const classes = Dict(zip(instances(EMNIST), [47, 62, 47, 10, 26, 10]))
const image_size = 28 ^ 2
const link = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
const scale = 20
const zip_path = joinpath(@__DIR__, "..", "emnist.zip")

function load_bytes(path::String, offset::Int)
    file = gzopen(path)
    data = deleteat!(read(file), 1:offset)
    close(file)
    data
end

load_input(path::String) = reshape(Float32.(load_bytes(path, 16)), image_size, :)

function load_labels(path::String, n::Int, is_numeric::Bool)
    bytes = load_bytes(path, 8)
    labels = zeros(Float32, n, length(bytes))

    for (i, label) in zip(bytes, eachcol(labels))
        label[i + is_numeric] += 1
    end

    labels
end

function load_file(f, files, ss)
    path = tempname()

    for file in files
        name = file.name

        if all(s -> contains(name, s), ss)
            open(_file -> print(_file, read(file, String)), path; write = true)
            break
        end
    end

    f(path)
end

load_emnist(emnist::EMNIST, is_numeric::Bool = emnist != letters) = get!(cache, emnist) do
    if !isfile(zip_path)
        @info "Downloading the EMNIST dataset"
        download(link, zip_path)
    end

    reader = Reader(zip_path)
    files = reader.files
    _split = "emnist-" * replace(string(emnist), '_' => "")
    n = classes[emnist]
    f = path -> load_labels(path, n, is_numeric)

    train_input = load_file(load_input, files, [_split, "train-images"])
    test_input = load_file(load_input, files, [_split, "test-images"])
    train_labels = load_file(f, files, [_split, "train-labels"])
    test_labels = load_file(f, files, [_split, "test-labels"])

    close(reader)

    (train_input, test_input, train_labels, test_labels) => n
end

identify(x) = argmax.(eachcol(x))

function render(image)
    Drawing(28 * scale, 28 * scale, "image.svg")

    for i in eachindex(image)
        column, row = scale .* divrem(i - 1, 28)
        setcolor(image[i] / 255 .* (1, 1, 1))
        box(Point(column, row), Point(column + scale, row + scale); action = :fill)
    end

    finish()
end

function test_emnist(emnist::EMNIST; parameters...)
    is_numeric = emnist != letters
    (train_input, test_input, train_labels, test_labels), n = load_emnist(emnist, is_numeric)

    nn = _train_classifier(train_input, train_labels, n; parameters...)

    for (image, prediction, label) in zip(
        eachcol(train_input), identify(nn(train_input)), identify(train_labels)
    )
        render(image)
        @show prediction label
        readline()
    end
end
