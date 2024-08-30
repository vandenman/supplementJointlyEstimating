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

analysis_files = joinpath.(pwd(), "data_analyses", (
    "multilevel_analysis.jl", "individual_analysis.jl", "aggregated_analysis.jl",
    "preprocess_results_data_analyses.jl", "figures_data_analyses.jl"
))

if any(!(isfile), analysis_files)

    missing_files = filter(!isfile, analysis_files)

    error("""
The following analysis script(s) were not found at these locations:

$(join('\t' .* missing_files, "\n"))

Please make sure that the working directory points to the root of the repository.
You can print the output of `pwd()` in julia to see what it currently is.

Right now the working directory is:

$(pwd())

""")

    !isinteractive() && exit(1)
end

# get the path to the currently running julia instance
path_julia_executable = joinpath(Sys.BINDIR, Base.julia_exename())

cmds = (
    ("Multilevel analysis", `$(path_julia_executable) --project=$(pwd()) --threads=$(no_threads_to_use) $(analysis_files[1]) test_run=$(test_run)`),
    ("Individual analysis", `$(path_julia_executable) --project=$(pwd()) --threads=$(no_threads_to_use) $(analysis_files[2]) test_run=$(test_run)`),
    ("Aggregated analysis", `$(path_julia_executable) --project=$(pwd()) --threads=$(no_threads_to_use) $(analysis_files[3]) test_run=$(test_run)`),
    ("Postprocessing",      `$(path_julia_executable) --project=$(pwd()) --threads=$(no_threads_to_use) $(analysis_files[4]) test_run=$(test_run)`),
    ("Creating figures",    `$(path_julia_executable) --project=$(pwd()) --threads=$(no_threads_to_use) $(analysis_files[5]) test_run=$(test_run)`)
)

for (name, cmd) in cmds
    log_message("Starting $(lowercase(name))")
    log_message("Running: $cmd")
    t0 = Dates.now()
    run(cmd)
    t1 = Dates.now()
    delta_t = Dates.canonicalize(t1 - t0)
    log_message("$name finished in $delta_t")
end
