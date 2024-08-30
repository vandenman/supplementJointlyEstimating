import CpuId, Dates, DelimitedFiles, StatsBase

function log_message(message)
    print("[")
    print(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
    print("] ")
    println(message)
end

# function parse_test_run_args()
#     isempty(ARGS) && return true
#     return first(ARGS) == "test_run=true"
# end

function get_no_threads_to_use()
    no_physical_cores = CpuId.cpucores()
    no_threads        = CpuId.cputhreads()

    no_threads_to_use = if no_threads == no_physical_cores
        no_physical_cores - 1
    else
        no_threads - (no_threads > 16 ? 4 : 2)
    end
    return no_threads_to_use
end

function is_test_run()
    if isempty(ARGS)
        log_message("Empty ARGS, assuming test run")
        return true
    end
    for arg in ARGS
        if '=' in arg
            key, value = split(arg, '=')
            if key == "test_run"
                returnval = value != "false"
                log_message("ARGS contains \"$arg\", continuing with $(returnval ? "test" : "real") run")
                return returnval
            end
        end
    end
    return true
end


read_file(file) = DelimitedFiles.readdlm(file, '\t', Float64)

function read_and_prepare_data(files::AbstractVector, p = nothing)

    rawdata = read_file(first(files))
    n = size(rawdata, 2)
    p = size(rawdata, 1)
    k = length(files)

    sum_of_squares = Array{Float64, 3}(undef, p, p, k)
    sum_of_squares[:, :, 1] = StatsBase.scattermat(rawdata[1:p, :]; dims=2)
    for i in 2:length(files)
        rawdata = read_file(files[i])[1:p, :]
        sum_of_squares[:, :, i] = StatsBase.scattermat(rawdata; dims=2)
    end

    return (; n, p, k, sum_of_squares)
end