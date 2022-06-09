
# TODO: clean up
# TODO: improve memory usage for datasets

# const DATASETS = ["mnist", "balanced", "digits", "letters", "bymerge", "byclass"]
const DIRECTORY = dirname(pwd()) * "/emnist/"
const FILE_NAMES = Dict(
    "balanced" => Dict(
        "mapping" => "emnist-balanced-mapping.txt",
        "test_images" => "emnist-balanced-test-images-idx3-ubyte.gz",
        "test_labels" => "emnist-balanced-test-labels-idx1-ubyte.gz",
        "train_images" => "emnist-balanced-train-images-idx3-ubyte.gz",
        "train_labels" => "emnist-balanced-train-labels-idx1-ubyte.gz"
    ),
    "byclass" => Dict(
        "mapping" => "emnist-byclass-mapping.txt",
        "test_images" => "emnist-byclass-test-images-idx3-ubyte.gz",
        "test_labels" => "emnist-byclass-test-labels-idx1-ubyte.gz",
        "train_images" => "emnist-byclass-train-images-idx3-ubyte.gz",
        "train_labels" => "emnist-byclass-train-labels-idx1-ubyte.gz"
    ),
    "bymerge" => Dict(
        "mapping" => "emnist-bymerge-mapping.txt",
        "test_images" => "emnist-bymerge-test-images-idx3-ubyte.gz",
        "test_labels" => "emnist-bymerge-test-labels-idx1-ubyte.gz",
        "train_images" => "emnist-bymerge-train-images-idx3-ubyte.gz",
        "train_labels" => "emnist-bymerge-train-labels-idx1-ubyte.gz"
    ),
    "digits" => Dict(
        "mapping" => "emnist-digits-mapping.txt",
        "test_images" => "emnist-digits-test-images-idx3-ubyte.gz",
        "test_labels" => "emnist-digits-test-labels-idx1-ubyte.gz",
        "train_images" => "emnist-digits-train-images-idx3-ubyte.gz",
        "train_labels" => "emnist-digits-train-labels-idx1-ubyte.gz"
    ),
    "letters" => Dict(
        "mapping" => "emnist-letters-mapping.txt",
        "test_images" => "emnist-letters-test-images-idx3-ubyte.gz",
        "test_labels" => "emnist-letters-test-labels-idx1-ubyte.gz",
        "train_images" => "emnist-letters-train-images-idx3-ubyte.gz",
        "train_labels" => "emnist-letters-train-labels-idx1-ubyte.gz"
    ),
    "mnist" => Dict(
        "mapping" => "emnist-mnist-mapping.txt",
        "test_images" => "emnist-mnist-test-images-idx3-ubyte.gz",
        "test_labels" => "emnist-mnist-test-labels-idx1-ubyte.gz",
        "train_images" => "emnist-mnist-train-images-idx3-ubyte.gz",
        "train_labels" => "emnist-mnist-train-labels-idx1-ubyte.gz"
    )
)

function read_string(file_name)
    file = open(file_name)
        data::Array{String, 1} = readlines(file)
    close(file)
    return data
end

function mapping(dataset)
    mapping::Dict{UInt8, Char} = Dict()

    for line in read_string(DIRECTORY * "gzip/" * FILE_NAMES[dataset]["mapping"])
        map = split(line, " ")
        mapping[parse(UInt8, map[begin]) + 1] = Char(parse(UInt8, map[end]))
    end

    return mapping
end

function read_uint8(file_name, offset)
    file = GZip.open(file_name)
        data::Array{UInt8, 1} = deleteat!(read(file), 1:offset)
    close(file)

    return convert.(Int32, data)
end

function read_labels(file_name, offset, dataset)
    iⱼ = read_uint8(file_name, offset) .+ 1

    # one-hot encoding
    yⱼ = zeros(length(mapping(dataset)), length(iⱼ))
    for (y, i) in zip(eachcol(yⱼ), iⱼ)
        y[i] += 1
    end

    return yⱼ
end

function read_images(file_name, offset)
    data = read_uint8(file_name, offset)
    return reshape(data, (28 ^ 2, :))
end

# function fix_letters_map!()
#     path = pwd() * "emnist/decompressed/" * "letters_mapping.txt"
#     mapping = read_string(path)

#     map::Array{Tuple{Int64, UInt8}, 1} = []
#     # why is key UInt8 and val Int?
#     for line in mapping
#         push!(map, (parse(UInt8, split(line, " ")[1]) - 1, parse(UInt8, split(line, " ")[2])))
#     end

#     f = open(path, "w")
#     for key_val in map
#         write(f, string(key_val[1]) * " " * string(key_val[2]) * "\n")
#     end
#     close(f)

#     return nothing
# end

# function fix_letters_labels!(key)
#     path = pwd() * "emnist/decompressed/letters_" * key * "_labels.bin"
#     labels = read_uint8(path, 0)
#     f = open(path, "w")
#     write(f, labels .- 1)
#     close(f)

#     return nothing
# end

function load_emnist(dataset)
    groups = ["mapping", "train_images", "test_images", "train_labels", "test_labels"]

    # check that the needed files are stored
    # if not, download and decompress them
    if !all(group -> isfile(DIRECTORY * "gzip/" * FILE_NAMES[dataset][group]), groups)
        rm(DIRECTORY * "gzip", force = true)
        mkpath(DIRECTORY * "gzip")
        emnist = "http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"

        if !isfile("emnist.zip")
            println("Downloading 'emnist.zip' from '" * emnist * "'.")
            download(emnist, DIRECTORY * "emnist.zip")
        end

        zip = ZipFile.Reader(DIRECTORY * "emnist.zip")

        for file in zip.files
            touch(DIRECTORY * file.name)

            open(DIRECTORY * file.name, "w") do io
                write(io, read(file, String))
            end
        end

        close(zip)
        rm(DIRECTORY * "emnist.zip", force = true)

        # TODO: fix this function
        # dataset == "letters" && fix_letters!()
    end

    x = read_images(DIRECTORY * "gzip/" * FILE_NAMES[dataset]["train_images"], 16)
    x = hcat(x, read_images(DIRECTORY * "gzip/" * FILE_NAMES[dataset]["test_images"], 16))
    y = read_labels(DIRECTORY * "gzip/" * FILE_NAMES[dataset]["train_labels"], 8, dataset)
    y = hcat(y, read_labels(DIRECTORY * "gzip/" * FILE_NAMES[dataset]["test_labels"], 8, dataset))

    return Data(x, y)
end
