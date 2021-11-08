# Supplementary code to "Optimal Search and Discovery" 

This Julia package replicates the results presented in Section 6.4 and produces simulations for a wider range of parameter values not presented in the paper.

Paper: [https://doi.org/10.1287/mnsc.2021.4085](https://doi.org/10.1287/mnsc.2021.4085https://doi.org/10.1287/mnsc.2021.4085)

## Dependencies
- Julia 1.5 (more recent versions should also not break things)
- Packages specified in "Project.toml" 

## Installation instructions
1. Extract files from this package into a folder (e.g. via `git clone https://github.com/rgreminger/SearchDiscoveryReplication.jl`)
2. In Julia, navigate to the folder (`cd("path/to/folder")`)
3. Activate environment by running `using Pkg; Pkg.activate(".")`
4. Install packages by running `Pkg.instantiate()` 

## Overview main files
- `replicate_main.jl`:		Produces results in Section 6.4 (results are saved in */gen*)
- `replicate_other_specs.jl`: Produces results for other parameter values (note that this can take some time) 
- `replicate_test_estim.jl`:	Checks whether models recover parameters when data is generated from self
- `src/replication.jl`:		Contains functions specific for replication

Documentation for the functions is provided through docstrings, which in Julia can be accessed through `?function` (e.g. `?runSimulations`). 

## Additional notes
Several core functions can use multi-threading for considerable performance improvements. To use this, start Julia with multiple threads enabled (see [Julia Docs](https://docs.julialang.org/en/v1/manual/multi-threading)).