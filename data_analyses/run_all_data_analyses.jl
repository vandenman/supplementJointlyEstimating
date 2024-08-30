include("../utilities.jl")

test_run = true

# this needs to be adjusted manually to whereever the data is stored
data_dir = "/home/don/hdd/surfdrive/Shared/GIN/"

if !isdir(data_dir) || isempty(readdir(data_dir))
    throw(error("Data directory \"$data_dir\" does not exist or is empty. Note that the raw data is only available upon request, and that the path to the data directory must be set manually in \"run_all_data_analyses.jl\"."))
    !isinteractive() && exit(1)
end

log_message("Running data analyses with test_run = $test_run")

no_threads_to_use = get_no_threads_to_use()

log_message("Running data analyses with $no_threads_to_use threads")

prefix = "julia --project=\".\" --threads=$(no_threads_to_use)"
postfix = "test_run=$test_run"
cmds = (
    multilevel = "$(prefix) $(joinpath("data_analyses", "multilevel_analysis.jl")) $postfix",
    individual = "$(prefix) $(joinpath("data_analyses", "individual_analysis.jl")) $postfix",
    aggregated = "$(prefix) $(joinpath("data_analyses", "aggregated_analysis.jl")) $postfix"
)

for (name, cmd) in pairs(cmds)
    log_message("Starting $name analysis")
    log_message("Running: $cmd")
    t0 = Dates.now()
    # run(`$cmd`)
    t1 = Dates.now()
    log_message("$name analysis finished in $(t1 - t0)")
end

if test_run
    log_message("Skipping postprocessing for test run")
    exit()
end

log_message("Starting postprocessing")
cmd = "$(prefix) $(joinpath("data_analyses", "preprocess_results_data_analyses.jl")) $postfix"
log_message("Running: $cmd")
t0 = Dates.now()
run(`$cmd`)
t1 = Dates.now()

log_message("Creating figures")
cmd = "$(prefix) $(joinpath("data_analyses", "figures_data_analyses.jl")) $postfix"
log_message("Running: $cmd")
t0 = Dates.now()
run(`$cmd`)
t1 = Dates.now()