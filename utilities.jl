import CpuId, Dates

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
                returnval = value != "true"
                log_message("ARGS contains \"$arg\", continuing with $(returnval ? "test" : "real") run")
                return returnval
            end
        end
    end
    return true
end