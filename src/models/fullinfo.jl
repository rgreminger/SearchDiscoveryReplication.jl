################################################################################
# Functions required to generate data based on different models

"""
	genData(m::Model,nCons::Int,nProdTot::Int,nProdCons::Int; fname::String = "" )

Returns a simulated dataset for specified model.  

### Arguments:
`m`:		Model specification. See `Model`. \\
`nCons`: 	# of consumers to simulate \\
`nProdTot`: 	# of alternatives in total (from which randomly chosen to display to consumer) \\
`nProdCons`: 	# of alternatives displayed to each consumer (incl. outside option). \\

### Optional keyword arguments: 
`fname`: 	Filename to save in BSON format (can use directory path).  \\
"""
function genData(m::FullInfo,nCons::Int,nProdTot::Int,nProdCons::Int; fname::String = "" )
	Random.seed!(m.seed)

	# Consumer index into full data 
	seli = [ 1+ii*nProdCons:(ii+1)*nProdCons for ii = 0:nCons-1]

	# Generate products 
	pid,chars = genProducts(m::Model,seli,nProdTot,nProdCons)

	# Gather data in NamedTuple for convenience
	d = (seli = seli,pid = pid,chars = chars)

	# Get purchase indicators 
	xβv = d.chars * m.β .+ rand(m.dV,size(d.chars,1)) 
	u   = xβv .+ rand(m.dE,size(d.chars,1))
	purch = genSearchPaths(m,d,u)
	
	# Gather again data 
	d = (seli = seli,pid = pid,chars = chars, purch = purch)

	if !isempty(fname)
		@info "Saved data in bson format to $fname."
		@save fname d
	end

	return d
end

###################################################
function genSearchPaths(m::FullInfo,d,u)
	n = length(d.seli)

	# Same for all consumers 
	purch = fill(0,n) 

  @inbounds Threads.@threads for k = 1:length(d.seli)
		for ii in getrange(n) # Loop over consumers allocated to specific thread 
			purch[ii] = genSearchPathi!(m::FullInfo,d.seli[ii],u)
		end
	end
	return purch
end

function genSearchPathi!(m::FullInfo,seli,u)
	_,purch = findmax_range(u,seli)
	return purch -seli[1] + 1  # adjustment as findmax_range gives index in whole array
end



###################################################
function negLogLik(m::FullInfo,β::Vector{T},d; e = [], v = [] ) where T<:Real

	# Draw random numbers if not provided 
	if isempty(e) 
		Random.seed!(m.seed+1)
		e = rand(m.dE,size(d.chars,1),m.options.nDraws)
	end
	if isempty(v)
		Random.seed!(m.seed+298)
		v = rand(m.dV,size(d.chars,1),m.options.nDraws)
	end

	# Form utility
	u = zeros(T,size(d.chars,1),m.options.nDraws) 
	@views fillU!(u,d,β,e,v) 

	# Loop through consumers (multi-threaded)
	logLik = zeros(eltype(β), Threads.nthreads())

	@inbounds Threads.@threads for k = 1:Threads.nthreads()
		local locSumLL = 0.0 
		for i in getrange(length(d.seli))
			r = d.seli[i] # selects consumers
			locV = @views calcVFI(u,r,r[d.purch[i]],m.smo) 
			locSumLL += log(locV)
		end
		logLik[k] = locSumLL
	end
	sum_logLik = abs(sum(logLik)) == Inf ? -1e10 : sum(logLik)
	return -sum_logLik
end

function fillU!(u,d,β,e,v) 
	nChar = size(d.chars,2)
	@inbounds Threads.@threads for i in 1:size(u,1)
		for k in 1: size(u,2) 
		u[i,k] = e[i,k] + v[i,k] 
			for j in 1:nChar 
				u[i,k] += d.chars[i,j] * β[j]
			end
		end
	end
end




function calcVFI(u,r,ind_p,smo)
	v = zero(eltype(u))
	n = size(u,2) 
	@inbounds for ee in 1:n
		v2 = zero(eltype(u))
		for j in r
				v2 += exp(-smo*(u[ind_p,ee]-u[j,ee]))
		end
		v += 1 / (1 + v2) / n  
	end
	return v 
end



###################################################

"""
	estimateModel(m::Model,d; startvals = [], pars_true = [])

Estimates model `m`. Returns tuple containing parameter estimates and full output from optimizer. 

### Arguments:
`m`:		Model specification. See `Model`.  \\
`d`: 		Data (see structure in a dataset generated through `genData`.) \\

### Optional keyword arguments: 
`startvals`: 	Vector of starting values for estimation. \\
`pars_true`: 	Only used for printing (useful when known).  \\
"""
function estimateModel(m::FullInfo,d; startvals = [] , pars_true = []  )

	# Preliminaries 	
	nBeta = size(d.chars,2) 

	if isempty(pars_true)
		pars_true = fill(NaN,nBeta)
	end

	if isempty(startvals)
		startvals = rand(Uniform(),nBeta) 
	end
	
	# Random draws, pre-allocate 
	Random.seed!(m.seed+651168) # ensures draws not same as in dataGen
	e = rand(m.dE,size(d.chars,1),m.options.nDraws)
	v = rand(m.dV,size(d.chars,1),m.options.nDraws)



		############################################################################
	# Optimization 

		# Define objective function 
	obj_fun(θ,p) = negLogLik(m,θ,d;v=v,e=e)
	obj = OptimizationFunction(obj_fun,m.options.autodiff)
	prob = OptimizationProblem(obj,startvals)

		# Run optimization 
	res = solve(prob,m.options.algo;
				show_trace = true , show_every=m.options.showEvery,
				maxiters = m.options.maxIter, 
				x_tol = m.options.x_tol,
				f_tol = m.options.f_tol,
				g_tol = m.options.g_tol)

	pars_opt = res.minimizer
	fval_opt = res.minimum
	
	# Calculate obj. fun at start and true val 
	startVal = obj_fun(startvals,[])
	trueVal = isnan(pars_true[1]) ? NaN : obj_fun(pars_true,[])

	# Extract results
	β = pars_opt
	finalVal = fval_opt

	infotxt = "
	####################################
	# Results of Estimation:
	Estimation model:	$(typeof(m))
	Final Objective Value: 		$finalVal
	Start Objective Value:		$startVal
	True Objective Value:		$trueVal
	Parameters:
		β  = $(round.(β,digits=2))
		β0 = $(round.(pars_true[1:nBeta],digits=2))
	#################################### "

	@info infotxt

	return β,finalVal

end

"""
	calcWelfareDemand(m::Model,d; e = [] , v = [] )

Simulates welfare and demand for a model given data. 

### Arguments:
`m`:		Model specification. See `Model`.  \\
`d`: 		Data (see structure in a dataset generated through `genData`.) 
### Optional keyword arguments: 
`e`: 	Matrix of ϵ error terms (draws along columns). \\
`v`: 	Matrix of ν error terms (draws along columns).   \\
"""
function calcWelfareDemand(m::FullInfo,d; e = [] , v = [] )

	# Draw random numbers if not provided 
	if isempty(e)
		Random.seed!(m.seed+43)
		e = rand(m.dE,size(d.chars,1),m.options.nDraws)
	end
	if isempty(v)
		Random.seed!(m.seed+323)
		v = rand(m.dV,size(d.chars,1),m.options.nDraws)
	end
	
	# Derived values 
	nCons = length(d.seli)

	# Form utility
	xβv = d.chars * m.β .+ v
	u   = xβv .+ e 

	# Pre allocate 
	sumU = zeros(Threads.nthreads())
	demand = zeros(Int64,length(d.pid[1]),Threads.nthreads())
	@inbounds Threads.@threads for k = 1:Threads.nthreads()	
		threadRange = getrange(length(d.seli))
		locU = 0.0 
		locD = zeros(Int64,length(d.pid[1]))
		for ii in threadRange, dd in 1:size(e,2)
			@views r 	= d.seli[ii]
			@views ud 	= u[:,dd]
			purchi 	= genSearchPathi!(m,r,ud)				
			locU 	+= ud[r[purchi]]
			locD[purchi] += 1
		end
		sumU[k] = locU / size(e,2)
		demand[:,k] .= locD 
	end
	return sum(sumU), sum(sumU), 0.0, sum(demand,dims=2) ./ size(e,2) ./ nCons
end