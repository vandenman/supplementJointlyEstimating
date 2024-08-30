include("../utilities.jl")

test_run = true

log_message("Running simulations with test_run = $test_run")

no_threads_to_use = get_no_threads_to_use()

log_message("Running simulations with $no_threads_to_use threads")

cmd_roc_simulation = "julia --project=\".\" --threads=$(no_threads_to_use) simulation_study/roc_simulation.jl test_run=$test_run"
cmd_mia_simulation = "julia --project=\".\" --threads=$(no_threads_to_use) simulation_study/multilevel_vs_individual_vs_aggregated.jl test_run=$test_run"

log_message("Starting ROC simulation")
log_message("Running: $cmd_roc_simulation")
t0 = Dates.now()
# run(`$cmd_roc_simulation`)
t1 = Dates.now()
log_message("ROC simulation finished in $(t1 - t0)")

log_message("Starting multilevel vs individual vs aggregated simulation")
log_message("Running: $cmd_mia_simulation")
t0 = Dates.now()
# run(`$cmd_mia_simulation`)
t1 = Dates.now()
log_message("multilevel vs individual vs aggregated simulation finished in $(t1 - t0)")
