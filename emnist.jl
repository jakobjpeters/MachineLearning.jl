
# TODO: clean up
# TODO: improve memory usage for datasets

const datasets = ["mnist", "balanced", "digits", "letters", "bymerge", "byclass"]
const dir = pwd() * "/emnist/"
const file_names = Dict(
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

function read_string(f_name)
    f = open(f_name)
    data::Array{String, 1} = readlines(f)
    close(f)
    return data
end

function mapping(dataset)
    mapping = read_string(dir * "decompressed/" * dataset * "_mapping.txt")
    map::Dict{UInt8, Char} = Dict()

    for line in mapping
        map[parse(UInt8, split(line, " ")[1]) + 1] = Char(parse(UInt8, split(line, " ")[2]))
    end

    return map
end

function read_uint8(f_name, offset)
    f = open(f_name)
    data::Array{UInt8, 1} = deleteat!(read(f), 1:offset)
    close(f)

    return convert.(Int64, data)
end

function read_labels(f_name, offset, dataset)
    indices = read_uint8(f_name, offset) .+ 1

    # one-hot encoding
    labels = zeros(length(mapping(dataset)), length(indices))
    for (col, i) in zip(eachcol(labels), indices)
        col[i] += 1
    end

    return labels
end

function read_images(f_name, offset)
    data = read_uint8(f_name, offset)
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

function decompress(in_name, out_name)
    f = GZip.open(in_name)
    write(out_name, read(f))
    close(f)

    return nothing
end

function assert_files(dataset)
    groups = ["mapping", "train_images", "test_images", "train_labels", "test_labels"]
    extension = group -> group == "mapping" ? ".txt" : ".bin"
    missing_decompressed = filter(group -> !isfile(dir * "decompressed/" * dataset * "_" * group * extension(group)), groups)

    if !isempty(missing_decompressed)
        mkpath(dir * "decompressed")
        if !all(group -> isfile(dir * "gzip/" * file_names[dataset][group]), missing_decompressed)
            rm(dir * "gzip", force = true)
            mkpath(dir * "gzip")
            emnist = "http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"
            if !isfile("emnist.zip")
                println("Downloading 'emnist.zip' from '" * emnist * "'.")
                download(emnist, "emnist/emnist.zip")
            end
            zip = ZipFile.Reader(dir * "emnist.zip")
            for f in zip.files
                touch(dir * f.name)
                out = open(dir * f.name, "w")
                write(out, read(f, String))
                close(out)
            end
            close(zip)
            rm(dir * "emnist.zip", force = true)

            # dataset == "letters" && fix_letters!()
        end
        println("Decompressing missing files.")
        map(group -> decompress(dir * "gzip/" * file_names[dataset][group], dir * "decompressed/" * dataset * "_" * group * extension(group)), missing_decompressed)
    end
end

function load_emnist(dataset)
    assert_files(dataset)

    inputs = read_images(dir * "decompressed/" * dataset * "_train_images.bin", 16)
    inputs = hcat(inputs, read_images(dir * "decompressed/" * dataset * "_test_images.bin", 16))
    labels = read_labels(dir * "decompressed/" * dataset * "_train_labels.bin", 8, dataset)
    labels = hcat(labels, read_labels(dir * "decompressed/" * dataset * "_test_labels.bin", 8, dataset))

    return Data(inputs, labels)
end
