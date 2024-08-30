include("../utilities.jl")

test_run = true

log_message("Running simulations with test_run = $test_run")

no_threads_to_use = get_no_threads_to_use()

log_message("Running simulations with $no_threads_to_use threads")

simulation_files = joinpath.(pwd(), "simulation_study", ("roc_simulation.jl", "multilevel_vs_individual_vs_aggregated.jl"))

if any(!(isfile), simulation_files)
    missing_files = filter(!isfile, simulation_files)

    error("""
The following simulation script(s) were not found at these locations:

$(join('\t' .* missing_files, "\n"))

Please make sure that the working directory points to the root of the repository.
You can print the output of `pwd()` in julia to see what it currently is.

Right now the working directory is:

$(pwd())

""")

    !isinteractive() && exit(1)
end

println(pwd())
println(isfile.(simulation_files))

# get the path to the currently running julia instance
path_julia_executable = joinpath(Sys.BINDIR, Base.julia_exename())

cmd_roc_simulation = `$(path_julia_executable) --project=$(pwd()) --threads=$(no_threads_to_use) $(simulation_files[1]) test_run=$test_run`
cmd_mia_simulation = `$(path_julia_executable) --project=$(pwd()) --threads=$(no_threads_to_use) $(simulation_files[2]) test_run=$test_run`

log_message("Starting ROC simulation")

log_message("Running: $cmd_roc_simulation")
t0 = Dates.now()
run(cmd_roc_simulation)
t1 = Dates.now()
delta_t = Dates.canonicalize(t1 - t0)
log_message("ROC simulation finished in $(delta_t)")

log_message("Starting multilevel vs individual vs aggregated simulation")
log_message("Running: $cmd_mia_simulation")

t0 = Dates.now()
run(cmd_mia_simulation)
t1 = Dates.now()
delta_t = Dates.canonicalize(t1 - t0)
log_message("multilevel vs individual vs aggregated simulation finished in $(delta_t)")
