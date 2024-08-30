# Supplementary files for the manuscript "Jointly Estimating Individual and Group Networks from fMRI Data"

## Installing dependencies

The programming language `julia` can be obtained from https://julialang.org/downloads/.

Once installed, navigate to the directory where this repository is located and start julia.
Next, press `]` to enter the julia package manager, then run
```julia-repl
pkg> activate . # . means the current directory
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

After installing all required packages, navigate to the directory where this repository is located.
To run all the simulations, run the following command in a terminal:
```bash
julia --project=. simulation_study/run_all_simulations.jl
```

To run all data analyses, To run all the simulations, run this command:
```bash
julia --project=. data_analayses/run_all_data_analyses.jl
```
Note that the data is only available upon request, so the last command will not work out of the box.

By default, both the simulation and the data analyses scripts will run in parallel using almost all available cores.
Modify the scripts directly to change the number of cores used (e.g., change the variable `no_threads_to_use`).
In addition, both the data analysis and the simulation study are run with `test = true` by default.
It is recommended to first run the scripts in test mode to check that everything is working as expected.
Afterward, run it with `test = false` to replicate the results in the manuscript.
Note that this will take about a day for the data analyses and multiple days for the simulations.