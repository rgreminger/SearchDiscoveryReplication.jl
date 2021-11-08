function genData(m::Weitzman,nCons::Int,nProdTot::Int,
					nProdCons::Int; fname::String = "" , zsMat = []  )
	Random.seed!(m.seed)
	# Initial checks
	@argcheck round((nProdCons-1-m.nA0)/m.nd)==(nProdCons-1-m.nA0)/m.nd  "Number of products (per consumer) does not match m.nA0 and s.nd combo."
	
	# Search cost specification  
	cfun = 	getCfun(m.cfun)

	# Consumer index into full data 
	seli = [ 1+ii*nProdCons:(ii+1)*nProdCons for ii = 0:nCons-1]

	# Products' positions (same for all consumers) 
	pos = fill([zeros(Int64,1+m.nA0) ; repeat(collect(Int64,1:(nProdCons-1-m.nA0)/m.nd),inner=m.nd)],nCons)

	# Generate products 
	pid,chars = genProducts(m::Model,seli,nProdTot,nProdCons)

	# Gather data in NamedTuple for convenience
	d = (seli = seli,pid = vcat(pid...),chars = chars,pos = vcat(pos...))

	# Get ξ value 
	ξ   = isempty(zsMat) ? 
			[calcZs(0.0,m.dE,cfun(m.cs + m.cd * pp))[1] for pp in 0:nProdCons] : 
			[calcZs(zsMat,cfun(m.cs + m.cd * pp)) for pp in 0:nProdCons]


	# Draw random shocks 
	v = rand(m.dV,size(d.chars,1)) 
	e   = rand(m.dE,size(d.chars,1))

	# Get utilities and search values 
	u = zeros(size(d.chars,1))
	zs = similar(u) 
	fillUZS!(u,zs,m,d,m.β,ξ,e,v)

	path,purch,search = genSearchPaths(m,d,nProdCons,zs,u)

	path = [path[path[:,i].>0,i] for i =1:nCons]
	
	# Gather again data 
	d = (seli = seli,pid = vcat(pid...),chars = chars,pos = vcat(pos...),path = path, purch = purch, search = search, u=u , zs = zs)

	if !isempty(fname)
		@info "Saved data in bson format to $fname."
		@save fname d
	end

	return d
end

###################################################
function genSearchPaths(m::Weitzman,d,maxProd,zs,u)
	n = length(d.seli)

	# Same for all consumers 
	path = zeros(Int64,maxProd,n)
	purch = fill(0,n) 
	search = fill(false,maxProd,n)


  @inbounds Threads.@threads for k = 1:Threads.nthreads()
		threadRange = getrange(n)
		locPathi 	= zeros(Int64,maxProd)
		locSearchi  = fill(true,maxProd)
		for ii in threadRange # Loop over consumers allocated to specific thread 
			# Select consumer 
			r = d.seli[ii]
			nProd = length(d.seli[ii])
			# Reset locals 
			locPathi .= 0
			locSearchi .= true ; locSearchi[1] = false  # outside option not searchable
			# generate search paths (updates locPathi/locSearchi and returns purchi)
			purchi = genSearchPathi!(m,locPathi,locSearchi,zs[r],u[r])
			# Fill in values 
			path[:,ii] = locPathi 
			purch[ii] = purchi
			search[:,ii] = locSearchi
		end

	end
	search .= search .== false
	return path,purch,search
end

###################################################
function genSearchPathi!(m::Weitzman,path,searchable,zs,u)
	
	# Initial status 
	cc_s = 0 
	ind_p = 1 # index of maximum utility (initially outside option)
	u_max = u[ind_p] # max utility
	# Generate search paths 
	while true
		# Current largest search value 
		val_s,ind_s = findmax_subsel(zs,searchable)
		
		if val_s > u_max # search 
			cc_s  += 1 
			path[cc_s] = ind_s 
			searchable[ind_s] = false 
			u_max = if u[ind_s] > u_max 
						ind_p = ind_s 
						u[ind_s]
					else
						u_max 
					end
		else # Purchase and end search
			purch = ind_p 
			return purch
			break # end with purchase
		end
	end
end


###################################################
function negLogLik(m::Weitzman,θ::Vector{T},nBeta,d,cfun,maxPos; 
					e = [], v = [],
					zsMat = [] ) where T<:Real

	# Extract Parameters
	β = θ[1:nBeta]
	cs = θ[nBeta+1]
	cd = if length(θ) > nBeta + 1 
			θ[nBeta+2]
		else
			zero(T)
	end

	# Draw random numbers if not provided 
	if isempty(e) 
		Random.seed!(m.seed+1)
		e = rand(m.dE,size(d.chars,1),m.options.nDraws)
	end
	if isempty(v)
		Random.seed!(m.seed+298)
		v = rand(m.dV,size(d.chars,1),m.options.nDraws)
	end

	u = zeros(T,size(d.chars,1),m.options.nDraws)
	zs = similar(u) 

	ξ   = isempty(zsMat) ? 
			[calcZs(0.0,m.dE,cfun(cs + cd * pp))[1] for pp in 0:maxPos] : 
			[calcZs(zsMat,cfun(cs + cd * pp)) for pp in 0:maxPos]

	fillUZS!(u,zs,m,d,β,ξ,e,v)
	
	# Loop through consumers
	logLik = zeros(T,Threads.nthreads())

	@inbounds Threads.@threads for k = 1:Threads.nthreads()
		locSumLL = zeros(T,1) 
		locV = zeros(T,3,size(e,2))
		locV2 = zeros(T,2)
		for ii in getrange(length(d.seli))
			# v1: must have continued searching until nth searched
			# v2: must have stopped after nth search
			# v3: must have chosen best out of consideration set
			locV .= 0.0
			fillV!(locV,d.path[ii],d.purch[ii],u,zs,d.seli[ii],m.smo)
			addLL!(locSumLL,locV,locV2)
		end
		logLik[k] = locSumLL[1]
	end
	sum_logLik = abs(sum(logLik)) == Inf ? -1e10 : sum(logLik)
	return -sum_logLik
end

function fillUZS!(u,zs,m::Weitzman,d,β,ξ,e,v)
	@inbounds Threads.@threads for i in 1:size(u,1)
		for j in 1:size(u,2) 
			u[i,j] = e[i,j] + v[i,j] 
			zs[i,j] = v[i,j]
			for k in 1:length(β)
				u[i,j] += d.chars[i,k] * β[k]
				zs[i,j] += d.chars[i,k] * β[k] 
			end
			zs[i,j] += ξ[d.pos[i]+1]
		end
	end
	return nothing 
end

function addLL!(locSumLL,locV,locV2)
	nDraws = size(locV,2) 
	locV2[2] = 0.0 
	@inbounds for e in 1:nDraws
		locV2[1] = 0.0
		for k in 1:size(locV,1) 
			locV2[1] += locV[k,e] 
		end
		locV2[2] += 1 /(1 + locV2[1])  / nDraws 
	end
	locSumLL[1] += log(locV2[2])
	return nothing 
end



#= 
Fills liklihood contributions for a given consumer into matrix `v`.

`v[1,:]`: Continuation rule: Reservation value of search in t needs to be larger than max utility of consideration set in t.

`v[2,:]`: Stopping rule: Max utility in consideration set at purchase in t needs to be larger than search values of all unsearched products in t.

`v[3,:]`: Purchase rule: Utility of purchased needs to be max in consideration set at time of purchase.
### Arguments:
`v`: 3 × nDraws array to be filled \\
`u`: nProd x nDraws array of product utilities of given consumer (u = xb + ϵ ). \\
`zs`: nProd x 1 array of search values for each product \\
`nSearch`: # searches \\
`u_p`: 1 x nDraws array of utility of product ultimately purchased by consumer \\
`smo`: Smoothing parameter
=# 
function fillV!(v,path,purch,u,zs,r,smo)
	nSearch = length(path)
	@inbounds for ee = 1:size(u,2) 
		u_p = u[r[purch],ee]
		if nSearch == 0  # no searches
			# v1 not defined, since no searches before
			v2!(v,u,zs,path,nSearch,r,ee,smo)
			# v3 not defined, since consideration set only contains outside option
		elseif nSearch == size(u,1)-1 # searched all products
			v1!(v,u,zs,path,nSearch,r,ee,smo)
			# v2 not defined, since no products left to search
			v3!(v,u_p,u,path,nSearch,r,ee,smo)
		else
			v1!(v,u,zs,path,nSearch,r,ee,smo)
			v2!(v,u,zs,path,nSearch,r,ee,smo)
			v3!(v,u_p,u,path,nSearch,r,ee,smo)
		end
	end
  	return
end

function v1!(v,u,zs,path,nSearch,r,ee,smo)
	um = u[r[1],ee] # outside option
	@inbounds for kk = 1:nSearch
		ip = r[path[kk]]
		v[1,ee] += exp(-smo*(zs[ip,ee] - um))
		um = max(um,u[ip,ee])
	end
end

function v2!(v,u,zs,path,nSearch,r,ee,smo)
	@inbounds for ll = 2:length(r)
		if ll in path
			continue
		end
		if nSearch > 0
			@views um = max(u[r[1],ee],findmax_range(u[r,ee],path)[1])
		else
			um = u[r[1],ee]
		end

		v[2,ee] += exp(-smo*(um-zs[r[ll],ee]))
	end
end

function v3!(v,u_p,u,path,nSearch,r,ee,smo)
	v[3,ee] += exp(-smo*(u_p-u[r[1],ee])) # outside option
	@inbounds for ll = 1:nSearch
		v[3,ee] += exp(-smo*(u_p-u[r[path[ll]],ee]))
	end
	v[3,ee] -= 1.0
end


###################################################
function estimateModel(m::Weitzman,d; startvals = [] , pars_true = [], fzsMat =[] )

	# Preliminaries 
	cfun = 	getCfun(m.cfun)
	
	nBeta = size(d.chars,2) 
	nCpar = if m.cd == 0 
				1 
			else
				2
			end
	
	if isempty(pars_true)
		pars_true = fill(NaN,nBeta+nCpar)
	end

	if isempty(startvals)
		startvals = rand(Uniform(),nBeta+nCpar) 
	end
	
	zsMat = isempty(fzsMat) ? [] : getIntExtPolZs(fzsMat)

	# Random draws, pre-allocate so not drawn in loglik function 
	Random.seed!(m.seed+651168) # ensures draws not same as in dataGen
	e = rand(m.dE,size(d.chars,1),m.options.nDraws)
	v = rand(m.dV,size(d.chars,1),m.options.nDraws)

	# Maximum Position 
	maxPos = maximum(d.pos)

	############################################################################
	# Optimization 
	# Define objective function 
	obj_fun(θ,p) = negLogLik(m,θ,nBeta,d,cfun,maxPos;v,e,zsMat)
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
	β = pars_opt[1:nBeta]
	cs = pars_opt[nBeta+1]
	if nCpar > 1
			cd = pars_opt[nBeta+2]
	end
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
		cs = $(round(cs,digits=2)) | cs0 = $(round(pars_true[nBeta+1],digits=2))"
	if nCpar >1 
		infotxt *= "
		cd = $(round(cd,digits=2)) | cd0 = $(round(pars_true[nBeta+2],digits=2)) " 
	end
	infotxt *= " 
	#################################### "
	

	@info infotxt

	if nCpar > 1 
		return β,cs,cd,finalVal
	else
		return β,cs,finalVal
	end

end

######################################################
function vecP(m::Weitzman)
	return vcat(m.β,m.cs,m.cd)
end


######################################################
function calcWelfareDemand(m::Weitzman,d; e = [] , v = [], zsMat = [])

	# Define values + cost function 
	cfun = getCfun(m.cfun)
	maxPos = maximum(d.pos)
	maxProd = maximum(length.(d.seli))
	nCons = length(d.seli)

	# Draw random numbers if not provided 
	if isempty(e) 
		Random.seed!(m.seed+1)
		e = rand(m.dE,size(d.chars,1),m.options.nDraws)
	end
	if isempty(v)
		Random.seed!(m.seed+298)
		v = rand(m.dV,size(d.chars,1),m.options.nDraws)
	end

	u = zeros(eltype(m.β),size(d.chars,1),size(e,2))
	zs = similar(u) 

	# position-specific costs and search values 
	cs = [cfun(m.cs + m.cd * pp) for pp in 0:maxPos]
	ξ   = isempty(zsMat) ? 
			[calcZs(0.0,m.dE,cs[i])[1] for i in eachindex(cs)] : 
			[calcZs(zsMat,cs[i]) for i in eachindex(cs)]

	fillUZS!(u,zs,m,d,m.β,ξ,e,v)

	# Pre allocate 
	sumU = zeros(Threads.nthreads())
	sumC = zeros(Threads.nthreads())
	demand = zeros(Int64,maxProd,Threads.nthreads())
	@inbounds Threads.@threads for k = 1:Threads.nthreads()	
		threadRange = getrange(nCons)
		locPathi 	= zeros(Int64,maxProd)
		locSearchi  = fill(true,maxProd)
		locSearchi[1] = false 
		
		locU = 0.0 
		locC = 0.0 
		locD = zeros(Int64,maxProd)
		for ii in threadRange
			r = d.seli[ii]
			@views posi = d.pos[r]
			for dd in 1:size(e,2)
				@views zsd 	= zs[r,dd]
				@views ud 	= u[r,dd]
				locPathi 	.= 0 
				locSearchi[2:end]  .= true
				purchi 	= genSearchPathi!(m,locPathi,locSearchi,zsd,ud)
				locU 	+= ud[purchi]
				for ss = 1:maxProd
					if locPathi[ss] == 0 
						break
					else
						locC += cs[posi[locPathi[ss]]+1]
					end
				end
				locD[purchi] += 1 
			end
		end
		sumU[k] = locU / size(e,2)
		sumC[k] = locC / size(e,2)
		demand[:,k] .= locD 
	end
	return sum(sumU) - sum(sumC), sum(sumU), sum(sumC), sum(demand,dims=2) ./ size(e,2) ./ nCons
end