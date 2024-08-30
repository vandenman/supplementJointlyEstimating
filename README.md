# Supplementary files for the manuscript "Jointly Estimating Individual and Group Networks from fMRI Data"

## Installing dependencies

Press `]` to enter the julia package manager, then run
```julia
pkg> activate .
pkg> instantiate
pkg> precompile
```
This will activate the project, download all the necessary packages, and precompile them.
A list of all the packages used in the project can be found in the `Manifest.toml` file.

If you want to use updated versions of the packages, you can run
```julia
pkg> update
```
but note that you will then use different packages than the ones used in the original project.

## Running the simulations and data analyses

To run all the simulations in a terminal, you can use the following command:
```bash
julia --project=. simulation_study/run_all_simulations.jl
```

To run all the data analyses in a terminal, you can use the following command:
```bash
julia --project=. data_analayses/run_all_data_analyses.jl
```
Note that the data is only available upon request, so the last command will not work out of the box.

By default, both the simulation and the data analyses scripts will run in parallel using almost all available cores.
In addition, both the simulations and data_analyses can be run in a "test" mode by setting `test = true`. Inside `run_all_xxx.jl`.
This will run the full pipeline but using fewer simulation conditions and posterior samples.
It is recommended to first run the scripts in test mode to check that everything is working as expected, and then with `test = false`, which will take about a day for the data analyses and multiple days for the simulations.