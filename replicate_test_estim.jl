using Distributions, SearchDiscoveryReplication, GalacticOptim, NLopt
using BSON: @save, @load
using PDMats # required for BSON to load data again 

##############################################
## Simulation for smoothing Weitzman1 (with ranking effects) 

# NOTE: Throughout uses default estimation options (see definitions_struct.jl)

# Set parameter values (same as baseline)
β0 = [1.0 , -1.0,3.5]
cs0 = 0.03
cd0 = 0.08
dChars = MvNormal([2., 3.5], [3.0, 1.0])
outDummy = true
dV = Normal(0, 0)
nCons = 2000
# By having many products in total, it is unlikely that the same product is shown to two  
# different consumers
nProdTot = 1000000 
nProdCons = 30 
nA0 = 1
seed = 23

# Create DS model 
W = Weitzman(
	β = β0,
	cs = cs0,
	cd = cd0, 
	nd = 1,
	nA0 = nA0,
	dChars = dChars,
	outDummy = outDummy,
	seed = seed ,
	dV = dV,
	dE = Normal(),
	cfun = "linear",
	smo = 10,
	options = estimOptions(algo=Opt(:LN_BOBYQA,length(β0)+2),maxIter = 1e8,f_tol = 1e-8, x_tol = 1e-8)
	)

# Generate data 
d = genData(W,nCons,nProdTot,nProdCons)

# Estimate and check results at end
res = estimateModel(W,d; startvals = vcat(β0,cs0,cd0),  pars_true = vcat(β0,cs0,cd0))

##############################################
## Simulation for smoothing Weitzman2 (no ranking)

# Create DS model 
Wf = Weitzman(
	β = β0,
	cs = cs0,
	cd = 0, 
	nd = 1,
	nA0 = nA0,
	dChars = dChars,
	outDummy = outDummy,
	seed = seed ,
	dV = dV,
	dE = Normal(),
	cfun = "linear",
	smo = 10,
	options = estimOptions(algo=Opt(:LN_BOBYQA,length(β0)+1),maxIter = 1e8,f_tol = 1e-8, x_tol = 1e-8)
	)

# Generate data 
d = genData(Wf,nCons,nProdTot,nProdCons)

# Estimate and check results at end
res = estimateModel(Wf,d; startvals = vcat(β0,cs0),  pars_true = vcat(β0,cs0))


##############################################
## Simulation for smoothing RS
RS = RandomSearch(
	β = β0,
	cs = cs0+cd0, 
	dChars = dChars,
	outDummy = outDummy,
	seed = seed,
	dV = dV,
	dE = Normal(),
	cfun = "linear",
	smo = 10,
	options = estimOptions(algo=Opt(:LN_BOBYQA,length(β0)+1))
	)
d = genData(RS,nCons,nProdTot,nProdCons)

# Estimate and check results at end (fct uses random starting values)
res = estimateModel(RS,d; startvals = vcat(β0,cs0+cd0),  pars_true = vcat(β0,cs0+cd0))


##############################################
## Simulation for smoothing FI
FI = FullInfo(
	β = β0,
	dChars = dChars,
	outDummy = outDummy,
	seed = seed + 1,
	dV = dV,
	dE = Normal(),
	smo = 10,
	options = estimOptions(algo=Opt(:LN_BOBYQA,length(β0)))
	)
d = genData(FI,nCons,nProdTot,nProdCons)
res = estimateModel(FI,d; startvals = β0,  pars_true = β0)

