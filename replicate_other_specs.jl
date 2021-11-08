using Distributions, SearchDiscoveryReplication, NLopt, GalacticOptim
using BSON
using PDMats # required for BSON to load data again 
##########################################################################
## Loop simulation over different parameter values 
function loopSimulation(βpVec,βoVec,csVec,cdVec)
	# Start at last index from filenames 
	fnames = readdir("./gen")
	# only keep relevant filenames
	fnames = fnames[(occursin.("results_spec",fnames)) .& 
				(occursin.("~",fnames) .== false) ]   
	i = maximum([convert(Int,parse(Float64,fnames[i][13:14])) for i in 1:length(fnames)]) 
	i += 1  
	
	# Loop over inputs
	for βp0 in βpVec, βo0 in βoVec, cs0 in csVec, cd0 in cdVec 
		β0 = [3.5,βp0,βo0] 

		d, wSD, outFI, outRS, outWf, outW = runSimulation(β0, cs0, cd0, dChars, outDummy, dV, 
															nCons, nProdTot, nProdCons,nA0;
															seed =26 +i,
															options = estimOptions(algo = Opt(:LN_BOBYQA,1))) 

		BSON.@save "gen/results_spec$i.bson" wSD outFI outRS outWf outW β0 cs0 cd0 dChars outDummy dV nA0 
		println(i)
		i += 1 
	end
end

cdVec = 0.01:0.05:0.25 
csVec = 0.01:0.05:0.15
βpVec = [-1.0]
βoVec = 1.0:5.0
dChars = MvNormal([2., 3.5], [3.0, 1.0])
outDummy = true
dV = Normal(0, 0)
nCons = 1000
nProdTot = 100000
nProdCons = 30 
nA0 = 1

loopSimulation(βpVec,βoVec,csVec,cdVec)

##########################################################################
## Summarize results (get same numbers as in tables)

function evaluateResultsLoop()
	fnames = readdir("./gen")
	fnames = fnames[(occursin.("results_spec",fnames)) .& (occursin.("~",fnames) .== false) ] # exclude unnecessary filenames

	# pre-allocate
	resSD = (zeros(7,length(fnames)),zeros(6,length(fnames)))
	resW = deepcopy(resSD)
	resWf = deepcopy(resSD)
	resRS = deepcopy(resSD)
	resFI = deepcopy(resSD)

	# Loop through results files, and fill in values for each of the models 
	for i in eachindex(fnames)
		println("Checking spec $(fnames[i])")
		estimOut, welfareChangesOut, demandOut, dStats = gatherResults("gen/$(fnames[i])")  
		for (r,m) in zip([resSD,resWf,resW,resRS,resFI],1:5)
			fillResults!(r,i,m,estimOut,welfareChangesOut,demandOut,dStats)
		end
	end
	return resSD,resWf,resW,resRS,resFI
end

function fillResults!(r,i,m,estimOut,welfareChangesOut,demandOut,dStats)

	# Same numbers as TABLE 2 
	r[1][1,i] = dStats[m,1] # n Searches
	r[1][2,i] = (1 - dStats[m,2]) * 100 # Share purchases 
	r[1][3,i] = estimOut[2,m] # price coefficient
	r[1][4:end,i] = estimOut[[1,3,4,5],m] / abs(estimOut[2,m])
	
	# Same numbers as TABLE 3
	r[2][1:3,i] = hcat(welfareChangesOut[1][m,2],demandOut[1][m,1],demandOut[2][m,1])
	r[2][4:6,i] = hcat(welfareChangesOut[1][m,3],demandOut[1][m,2],demandOut[2][m,2])
end

# resSD,resWf,resW,resRS,resFI = evaluateResultsLoop()
res = evaluateResultsLoop()
##########################################################################
## Plot results (for easy comparison)
using Plots

ns = length(res[1][1][1,:])
nm = length(res) 
l = ["SD" "DS1" "DS2" "RS" "FI"]

# Table 2 numbers 
for k in 1:7
	local d = zeros(ns,nm) 
	for (i,r) in enumerate(res)
		d[:,i] = r[1][k,:]
	end
	p = plot(d, label = l, title = "Table 2: Column $k") 
	png(p,"gen/fig$(k)_for_t2.png")
end

# Table 3 numbers 
for k in 1:6
	local d = zeros(ns,nm) 
	for (i,r) in enumerate(res)
		d[:,i] = r[2][k,:]
	end
	p = plot(d, label = l, title = "Table 3: Column $k") 
	png(p,"gen/fig$(k)_for_t3.png")
end
