"""
	runSimulation(β0,cs0,cd0,dChars,
				outDummy,dV,nCons,nProdTot,nProdCons,nA0;
				seed = 24)

Runs simulation given parameter values. Returns the generated dataset and the results of FI, SO and W estimation. 

# Arguments:
- `β0`, `cs0`, `cd0`: True parameter values with which data is generated in SD model
- `dChars`:		Distribution of characteristics in X
- `outDummy`:   Bool whether to include a dummy for the outside option (if true, first element in β0 is preference parameter for it)
- `dV`:		 	Distribution of errors v. If all zeros, set to `Normal(0,0)`.
- `nCons`:		# consumers to simulate 
- `nProdTot`:   # of total products (from which a random subsample is shown to each consumer)
- `nProdCons`:  # of products shown to each consumer 
- `nA0`:		Size of initial awareness (if nA0 = 2, all consumers initially observe the first two products without having to discover them. But they still have to inspect them.)

# Optional arguments:
- `seed`: Sets seed throughout 
- `solver`: Option to set solver through either `Optim` or `NLopt` package. Array with first element string indicating package, and second element algorithm (e.g. `["Optim",Optim.NelderMead()"]` or `["NLopt",:LN_BOBYQA]`)
"""
function runSimulation(
	β0,
	cs0,
	cd0,
	dChars,
	outDummy,
	dV,
	nCons,
	nProdTot,
	nProdCons,
	nA0;
	seed = 24,
	 # uses default estimation options (defined in definitions_struct.jl)
	options = estimOptions()
)   

	###########
	# SD model 
	SD = SearchDiscovery(
		β = β0,
		cs = cs0,
		cd = cd0,
		dze = 0.0,
		dChars = dChars,
		outDummy = outDummy,
		seed = seed,
		nd = 1,
		nA0 = nA0,
		dV = dV,
		dE = Normal(),
		cfun = "linear",
		zdfun = "linear"
	)
	# Generate data 
	d = genData(SD, nCons, nProdTot, nProdCons)

	# # Calculate welfare changes 
	wSD = calcWelfareChanges(SD,d; seed = seed) 
	
	###########
	# Full information model 
	FI = FullInfo(
		β = β0,
		dChars = dChars,
		outDummy = outDummy,
		seed = seed + 1,
		dV = dV,
		dE = Normal(),
		smo = 3,
		options=options
	)
	# For NLopt algorithms, replace here since need number of parameters
	if typeof(options.algo) <: Opt
		FI.options.algo = Opt(options.algo.algorithm,length(β0))
	end

	resFI = estimateModel(FI, d, startvals = β0)
	# input estimated parameters
	FI.β = resFI[1]
	# Calculate welfare changes 
	wFI = calcWelfareChanges(FI,d; seed = seed+4) 

	# Calcualte summary stats 
	dFI = calcSummaryStats(d,FI) 

	###########
	# Estimate search in order model 
	RS = RandomSearch(
		β = β0,
		cs = cs0, 
		dChars = dChars,
		outDummy = outDummy,
		seed = seed + 2,
		dV = dV,
		dE = Normal(),
		cfun = "linear",
		smo = 10,
		options=options,
	)
	if typeof(options.algo) <: Opt
		RS.options.algo = Opt(options.algo.algorithm,length(β0)+1)
	end

	resRS = estimateModel(RS, d, startvals = vcat(β0, cs0))
	
	# Input estimated pars 
	RS.β = resRS[1]
	RS.cs = resRS[2]
	# Calculate welfare changes 
	wRS = calcWelfareChanges(RS,d; seed = seed+4) 

	# Calcualte summary stats 
	dRS = calcSummaryStats(d,RS) 

	###########
	#  Estimate Weitzman model without ranking effects 
	Wf = Weitzman(
		β = β0,
		cs = cs0,
		cd = 0, 
		nd = 1,
		nA0 = nA0,
		dChars = dChars,
		outDummy = outDummy,
		seed = seed + 3,
		dV = dV,
		dE = Normal(),
		cfun = "linear",
		smo = 10,
		options=options,
	)
	if typeof(options.algo) <: Opt
		Wf.options.algo = Opt(options.algo.algorithm,length(β0)+1)
	end
	# Change structure of data 
	# d.pos / d.pid data different in Weitzman
	d1 = (seli=d.seli,pid = vcat(d.pid...), pos = vcat(d.pos...), 
			chars = d.chars, path = d.path, purch = d.purch)

	# Run estimation
	resWf = estimateModel(Wf, d1, startvals = vcat(β0, cs0))
	
	# Input estimated pars 
	Wf.β = resWf[1]
	Wf.cs = resWf[2]
	# Calculate welfare changes 
	wWf = calcWelfareChanges(Wf,d; seed = seed+4) 

	# Calcualte summary stats 
	dWf = calcSummaryStats(d,Wf) 

		###########
	#  Estimate Weitzman model with ranking effects 
	W = Weitzman(
		β = β0,
		cs = cs0,
		cd = cd0, 
		nd = 1,
		nA0 = nA0,
		dChars = dChars,
		outDummy = outDummy,
		seed = seed + 3,
		dV = dV,
		dE = Normal(),
		cfun = "linear",
		smo = 10,
		options=options,
	)
	if typeof(options.algo) <: Opt
		W.options.algo = Opt(options.algo.algorithm,length(β0)+2)
	end
	# Change structure of data 
	# d.pos / d.pid data different in Weitzman
	d1 = (seli=d.seli,pid = vcat(d.pid...), pos = vcat(d.pos...), 
			chars = d.chars, path = d.path, purch = d.purch)

	# Run estimation
	resW = estimateModel(W, d1, startvals = vcat(β0, cs0,cd0))
	
	# Input estimated pars 
	W.β = resW[1]
	W.cs = resW[2]
	W.cd = resW[3]
	# Calculate welfare changes 
	wW = calcWelfareChanges(W,d; seed = seed+4) 

	# Calcualte summary stats 
	dW = calcSummaryStats(d,W) 

	return d, wSD , (resFI,wFI,dFI),  (resRS,wRS,dRS), (resWf,wWf,dWf), (resW,wW,dW)
end

"""
	calcWelfareChanges(m::Model, d; seed = 511)
Calculates the welfare changes specified in the paper for a given model specification. Returns matrix with different changes in rows, and different measures in cols. 

*Rows*: 1. Baseline welfare | 2. No search and discovery costs | 3. 1% 'price' decrease of product on 5th position | 4. 1% 'price' decrase of product on 15th position 

*Columns*: 1. Welfare | 2. Utility | 3. Costs | 4. Demand Outside | 5. Demand Position 1 | 6. Demand Position 2 ....

# Arguments: 
- `m`: `Model` definition
- `d`: Data input

# Optional arguments: 
- `seed`: Sets seed for random draws in welfare calculation
	
"""
function calcWelfareChanges(m::Model, d; seed = 511)
	nCons = length(d.seli)
	# Make draws (stays same across changes)
	Random.seed!(seed)
	e = rand(m.dE, size(d.chars, 1), m.options.nDraws*5) # use more draws than in estimation for precision
	v = rand(m.dV, size(d.chars, 1), m.options.nDraws*5)

	d0 = 	if typeof(m) <: Weitzman # Changes structure of data for Weitzman model 
				(seli=d.seli,pid = vcat(d.pid...), pos = vcat(d.pos...), 
					chars = d.chars, path = d.path, purch = d.purch)
			else 
				d 
			end

	# Initial welfare 
	w0 = calcWelfareDemand(m, d0; e , v)

	# Create array to store values 
	out = fill(NaN,4,3+length(w0[4])) 

	# Fill in values from welfare 
	fillOut!(out,w0,1) 

	##############################
	# Remove all search costs  
	mn = FullInfo(β=m.β,dChars=m.dChars,outDummy=m.outDummy,
					dV=m.dV,dE=m.dE,smo=0) #smoothing does not matter here

	w1 = calcWelfareDemand(mn, d; e , v)
	fillOut!(out,w1,2) 

	##############################
	# 1% 'price' change of product shown on 5th position (for all consumers)
	dn = deepcopy(d0)
	iProd = findfirst( x-> x==5,d.pos[1])

	for ii in eachindex(dn.seli)
		if m.outDummy
			dn.chars[dn.seli[ii][iProd],end-1] = dn.chars[dn.seli[ii][iProd],end-1] * 0.99
		else
			dn.chars[dn.seli[ii][iProd],end] = dn.chars[dn.seli[ii][iProd],end] * 0.99
		end
	end

	w1 = calcWelfareDemand(m, dn; e , v)
	fillOut!(out,w1,3) 

	##############################
	# 1% 'price' decrease of product shown on 15th position (for all consumers)
	dn = deepcopy(d0)
	iProd = findfirst( x-> x == 15,d.pos[1])

	for ii in eachindex(dn.seli)
		if m.outDummy
			dn.chars[dn.seli[ii][iProd],end-1] = dn.chars[dn.seli[ii][iProd],end-1] * 0.99
		else
			dn.chars[dn.seli[ii][iProd],end] = dn.chars[dn.seli[ii][iProd],end] * 0.99
		end
	end

	w1 = calcWelfareDemand(m, dn; e , v)
	fillOut!(out,w1,4) 

	return out 
end

"""
	calcSummaryStats(d,m::Model) 
"""
function calcSummaryStats(d,m::Model) 
	Random.seed!(m.seed + 2934)
	nCons = length(d.seli) 

	nS = zeros(2) # Avg. number of searches 
	shareOutside = zeros(2) # Share outside option 

	# Calculate summary statistics in data 
	nS[1] = mean(length.(d.path))
	shareOutside[1] = count(d.purch .== 1) / nCons 

	# Simulate summary statistics (across m.options.nDraws draws)
	nSsim, shareOutsidesim = _calcSummaryStats(d,m) 
	nS[2] = nSsim 
	shareOutside[2] = shareOutsidesim

	return nS, shareOutside 
end


function _calcSummaryStats(d,m::FullInfo) 
	Random.seed!(m.seed+120937)
	nCons = length(d.seli) 

	nS = NaN 
	shareOutside = 0.0 
	xβv = zeros(size(d.chars,1))
	u 	= similar(xβv)
	for e in 1:m.options.nDraws
		xβv = d.chars * m.β .+ rand(m.dV,size(d.chars,1)) 
		u   = xβv .+ rand(m.dE,size(d.chars,1))
		purch = genSearchPaths(m,d,u)
		shareOutside += count(purch .== 1) / nCons / m.options.nDraws 
	end

	return nS, shareOutside
end


function _calcSummaryStats(d,m::RandomSearch) 
	Random.seed!(m.seed+120937)
	nCons = length(d.seli) 
	nProdCons = length(d.seli[1])

	# Get values need to generate search paths 
	cfun = getCfun(m.cfun)
	μ  = sum(mean(m.dChars) .* m.β[1:end-m.outDummy]) + mean(m.dV)
	σ2  = sum(var(m.dChars) .* m.β[1:end-m.outDummy] .^2) + var(m.dV)
	zs = calcZs(0.0,Normal(μ,sqrt(σ2)),cfun(m.cs))[1]

	xβv = zeros(size(d.chars,1))
	u 	= similar(xβv)

	nS = 0.0 
	shareOutside = 0.0 

	for e in 1:m.options.nDraws
		xβv = d.chars * m.β .+ rand(m.dV,size(d.chars,1)) 
		u   = xβv .+ rand(m.dE,size(d.chars,1))
		path,purch = genSearchPaths(m,d,nProdCons,zs,u)
		nS += mean(length.(path)) / m.options.nDraws 
		shareOutside += count(purch .== 1) / nCons / m.options.nDraws 
	end

	return nS, shareOutside

end


function _calcSummaryStats(d,m::Weitzman) 
	Random.seed!(m.seed+120937)
	nCons = length(d.seli) 
	nProdCons = length(d.seli[1])

	# Change data structure for Weitzman model 
	d1 = (seli=d.seli,pid = vcat(d.pid...), pos = vcat(d.pos...), 
				chars = d.chars, path = d.path, purch = d.purch)
	
	# Get values need to generate search paths 
	cfun = 	getCfun(m.cfun)
	ξ   = 	[calcZs(0.0,m.dE,cfun(m.cs + m.cd * pp))[1] for pp in 0:nProdCons]

	# Pre-allocate
	u = zeros(size(d.chars,1))
	zs = similar(u) 
	e = similar(u) 
	v = similar(u) 

	nS = 0.0 
	shareOutside = 0.0 

	for k in 1:m.options.nDraws
		rand!(m.dV,v) 
		rand!(m.dE,e) 
		fillUZS!(u,zs,m,d1,m.β,ξ,e,v)

		path,purch,_ = genSearchPaths(m,d1,nProdCons,zs,u)
		nS +=  countSearches(m,path) / nCons / m.options.nDraws 
		shareOutside += count(purch .== 1) / nCons / m.options.nDraws 
	end

	return nS, shareOutside 
end

function countSearches(m::Weitzman,path)
	nS = 0 
	for i in 1:size(path,2)
		_nS = findfirst( x -> x==0, path[:,i])
		nS += isnothing(_nS) ? length(path) -1  : _nS -1 
	end
	return nS 
end

"""
	gatherResults(β0,cs0,cd0, wSD,outFI,outRS,outW)
Gather results for table and analysis.
"""
function gatherResults(fname::String)	
	@load fname wSD outFI outRS outWf outW β0 cs0 cd0
	res = [outWf,outW,outRS,outFI] 

	##########################
	# Estimated parameters
	b = [m[1][1] for m in res] 
	b = hcat(b...)
	b = hcat(β0,b)

	cs = [m[1][2] for m in res[1:end-1]]
	cs = hcat(cs0,cs',NaN)

	cd = hcat(cd0,NaN,outW[1][3],NaN,NaN)

	L = [NaN]
	for m in res
		push!(L,-m[1][end])
	end
	estimOut = vcat(b,cs,cd,L')

	##########################
	# Welfare changes
	dwU = hcat(wSD[1,2],diffFirst(wSD)[:,2]')
	dwC = hcat(wSD[1,3],diffFirst(wSD)[:,3]')
	dw  = dwU .- dwC
	for m in res 
		wdiffd = diffFirst(m[2])
		dwU = vcat(dwU,hcat(m[2][1,2],wdiffd[:,2]'))
		dwC = vcat(dwC,hcat(m[2][1,3],wdiffd[:,3]'))
		dw  = vcat(dw,dwU[end,:]' .- dwC[end,:]')
	end
	# Normalize by second coefficient  (only matters for estimated welfare, not changes)
	for i = 1:size(dw,1), m in [dw,dwU,dwC]
		m[i,:] .= m[i,:] ./ abs(b[2,i])
	end

	# Put into % changes of initial welfare 
	for m in (dw,dwU,dwC)
		m[:,2:end] .= m[:,2:end] ./ abs.(m[:,1]) * 100
	end

	welfareOut = (dw,dwU,dwC)

	##########################
	# Demand changes
	dDemand5 = wSD[:,10]'
	dDemand1 = wSD[:,5]'
	for r in res 
		dDemand5 = vcat(dDemand5,r[2][:,10]')
		dDemand1 = vcat(dDemand1,r[2][:,5]') 
	end

	# Make into percentage changes 
	dDemand5 = (dDemand5[:,2:end] .- dDemand5[:,1] ) ./ dDemand5[:,1] .* 100 
	dDemand1 = (dDemand1[:,2:end] .- dDemand1[:,1] ) ./ dDemand1[:,1] .* 100 

	demandOut = (dDemand1,dDemand5)

	##########################
	# Extract size of search set and share of outside Option 

	dStats = [outFI[3][1][1] outFI[3][2][1]]  # Note, first element always from generated data
	for r in res
		dStats = vcat(dStats,[r[3][1][2] r[3][2][2]])  
	end

	return estimOut, welfareOut, demandOut, dStats 
end

##############################################################

function fillOut!(out,w,k) 
	for i in 1:size(out,2)
		out[k,i] = 	if i >= 4 
						w[4][i-3] # demand for each product (1. is outside)
					else
						w[i]
					end
	end
end

"""
	diffFirst(a::Array{T,2}) where T<:Real
Takes difference to first row for each row in array `a`. 
"""
function diffFirst(a::Array{T,2}) where T<:Real
	return hcat([a[i,:] .- a[1,:] for i = 2:size(a,1)]...)'
end