# Estimation/Simulation options, part of each model 
@with_kw mutable struct estimOptions
	algo = NelderMead()
	innerAlgo = NLopt.LN_BOBYQA # used only for MultiStart
	lb = nothing
	ub = nothing 
	autodiff = GalacticOptim.AutoFiniteDiff(;fdtype = Val(:central),fdhtype = Val(:hcentral))
	nDraws::Int64 = 500
	useHalton::Bool = false 
	showEvery::Int64 = 5 
	maxIter::Int64 = 100_000
	f_tol::Float64 = 0.0
	x_tol::Float64 = 0.0
	g_tol::Float64 = 1e-8
end


# Define abstract type, where each model is subtype of this 
abstract type Model end

#################################################
# Model definitions
# Uses Parameter.jl pkg which allows to set default values 
@with_kw mutable struct Weitzman <: Model 
	β::Array{Float64,1} 
	cs::Float64
	cd::Float64 
	nd::Int64 
	nA0::Int64  
	dChars::Distribution 
	outDummy::Bool
	dV::Normal			= Normal()
	dE::Normal			= Normal()
	cfun::String 		= "linear"
	seed::Int64 = 23909
	smo::Float64 
	options::estimOptions = estimOptions()
end 

@with_kw mutable struct FullInfo <: Model 
	β::Array{Float64,1} 
	dChars::Distribution 
	outDummy::Bool
	dV::Normal
	dE::Normal
	seed::Int64 = 23909
	smo::Float64
	options::estimOptions = estimOptions()
end 


@with_kw mutable struct RandomSearch <: Model 
	β::Array{Float64,1} 
	cs::Float64
	dChars::Distribution 
	outDummy::Bool
	dV::Normal
	dE::Normal
	cfun::String 
	seed::Int64 = 23909
	smo::Float64
	options::estimOptions = estimOptions()
end 


@with_kw mutable struct SearchDiscovery <: Model 
	β::Array{Float64,1} 
	cs::Float64
	cd::Float64 
	dze::Float64
	nd::Int64
	nA0::Int64 = 0 
	dChars::Distribution 
	outDummy::Bool
	dV::Normal
	dE::Normal
	cfun::String 
	zdfun::String
	seed::Int64 = 23909
	options::estimOptions = estimOptions()
end 

