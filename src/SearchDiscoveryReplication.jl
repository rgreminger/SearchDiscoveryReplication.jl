module SearchDiscoveryReplication
__precompile__()

# Exported functions and types 
export Model,estimOptions  # Structs
export Weitzman, FullInfo, RandomSearch, SearchDiscovery # Models 
export calcZd, calcZs # Reservation value calculation
export genData, estimateModel, calcWelfareDemand # Main functions for each model 

export runSimulation, calcWelfareChanges, gatherResults  # Functions specific for replication
export writeLatexTable # To write to latex table 

# Support pkgs
using Parameters: @with_kw
using ArgCheck: @argcheck
using Printf: @sprintf
# Math / Stats pkgs
using Distributions, QuadGK, Roots, ForwardDiff , StatsBase, Random, LinearAlgebra
# Optimizers
using NLopt, GalacticOptim
# Data org pkgs
using BSON: @save, @load 

include("definitions_struct.jl") # Own types 
include("bivariate.jl") # Functions for Bivariate Normal distribution 
include("reservation_values.jl") # Functions for reservation value calculation
include("support.jl")
# Different models 
include("models/weitzman.jl")
include("models/randomsearch.jl")
include("models/fullinfo.jl")
include("models/discovery_incomplete.jl")
# Specific for replication
include("replication.jl")

end
