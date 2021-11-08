function genData(m::RandomSearch,nCons::Int,nProdTot::Int,nProdCons::Int; fname::String = "" )
	Random.seed!(m.seed)

	# Consumer index into full data 
	seli = [ 1+ii*nProdCons:(ii+1)*nProdCons for ii = 0:nCons-1]

	# Generate products 
	pid,chars = genProducts(m::Model,seli,nProdTot,nProdCons)

	# Products' positions (same for all consumers) 
	pos = fill(collect(0:nProdCons-1),nCons)

	# Gather in table for convenience
	d = (seli = seli,pid = pid,chars = chars)

	# Get reservation value of continuing search
	cfun = getCfun(m.cfun)
	μ  = sum(mean(m.dChars) .* m.β[1:end-m.outDummy]) + mean(m.dV)
	σ2  = sum(var(m.dChars) .* m.β[1:end-m.outDummy] .^2) + var(m.dV)
	zs = calcZs(0.0,Normal(μ,sqrt(σ2)),cfun(m.cs))[1]
	
	xβv = chars * m.β .+ rand(m.dV,size(chars,1)) 
	u   = xβv .+ rand(m.dE,size(d.chars,1))
	path,purch = genSearchPaths(m,d,nProdCons,zs,u)

	path = [path[path[:,i].>0,i] for i =1:nCons]
	
	# Gather again data 
	d = (seli = seli,pid = pid,chars = chars,pos = pos,path = path, purch = purch, u=u , zs = zs)


	if !isempty(fname)
		@info "Saved data in bson format to $fname."
		@save fname d
	end

	return d
end

###################################################
function genSearchPaths(m::RandomSearch,d,maxProd,zs,u)
	n = length(d.seli)

	# Same for all consumers 
	path = zeros(Int64,maxProd,n)
	purch = fill(0,n) 

  @inbounds Threads.@threads for k = 1:Threads.nthreads()
		threadRange = getrange(n)
		locPathi 	= zeros(Int64,maxProd)
		for ii in threadRange # Loop over consumers allocated to specific thread 
			# Select consumer 
			r = d.seli[ii]
			nProd = length(d.seli[ii])
			# Reset locals 
			locPathi .= 0
			# generate search paths (updates locPathi/locSearchi and returns purchi)
			purchi = genSearchPathi!(m,locPathi,zs,u[r])
			# Fill in values 
			path[:,ii] = locPathi 
			purch[ii] = purchi
		end

	end
	return path,purch
end


###################################################
function genSearchPathi!(m::RandomSearch,path,zs,u)
# NOTE: ASSUMES TOP DOWN (which is same as random if Xs are random)
	# Initial status 
	cc_s = 1 
	ind_p = 1 # index of maximum utility (initially outside option)
	u_max = u[ind_p] # max utility
	# Generate search paths 
	while true		
		if cc_s == size(u,1) # Searched all 
			purch = findmax(u)[2]
			return purch			
		elseif zs > u_max # search 
			cc_s  += 1 
			path[cc_s] = cc_s 
			u_max = if u[cc_s] > u_max 
						ind_p = cc_s 
						u[cc_s]
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
function negLogLik(m::RandomSearch,θ::Vector{T},d,cfun; e = [], v = [] ) where T<:Real
	
	# Extract Parameters
	β = θ[1:end-1]
	cs = θ[end]

	# Draw random numbers if not provided 
	if isempty(e) 
		Random.seed!(m.seed+23)
		e = rand(m.dE,size(d.chars,1),m.options.nDraws)
	end
	if isempty(v)
		Random.seed!(m.seed+3598)
		v = rand(m.dV,size(d.chars,1),m.options.nDraws)
	end
	
	# Form utility
	xβv = d.chars * β .+ v
	u = xβv .+ e

	# Search values 
	μ  = sum(mean(m.dChars) .* m.β[1:end-m.outDummy]) + mean(m.dV)
	σ2  = sum(var(m.dChars) .* m.β[1:end-m.outDummy] .^2) + var(m.dV)
	zs = calcZs(0.0,Normal(μ,sqrt(σ2)),cfun(m.cs))[1]
	

	# Loop through consumers
	logLik = fill(0.0,Threads.nthreads())

	@inbounds Threads.@threads for k = 1:Threads.nthreads()
		locSumLL = 0.0 
		locV = zeros(Float64,3,size(e,2))
		for ii in getrange(length(d.seli))
			locV .= 0.0
			fillV_RS!(locV,d.path[ii],d.purch[ii],u,zs,d.seli[ii],m.smo)
			locSumLL += log(mean(1.0 ./ (1.0 .+ sum(locV,dims=1))))
		end
		logLik[k] = locSumLL
	end
	sum_logLik = abs(sum(logLik)) == Inf ? -1e10 : sum(logLik)
	return -sum_logLik
end

## 
# ind_p is index of purchased
function fillV_RS!(v,path,ind_p,u,zs,r,smo)
	nSearch = length(path)
	# Continuation, requires u <= reservation value for those where continued
	if nSearch > 0
		for ee = 1:size(u,2)
			# um = u[r[1],ee]
			# v[1,ee] += exp(-smo*(zs - um))
			# for ll in path[1:end-1]
			# 	um = max(um,u[r[ll],ee])
			# 	v[1,ee] += exp(-smo*(zs - um))
			# end
			u0 = u[r[1],ee]
			um = nSearch > 1 ? max(u0,findmax_range(u,r[path[1:end-1]],ee)[1],u0) : u0 
			v[1,ee] += exp(-smo*(zs - um))
		end
	end

	# Stopping, requires max util of searched > reservation value
	if nSearch < length(r)-1
		for ee = 1:size(u,2)
			u0 = u[r[1],ee]
			um = nSearch > 0 ? max(u0,findmax_range(u,r[path],ee)[1]) : u0
			v[2,ee] += exp(-smo*(um-zs))
		end
	end

	# Purchase requires being maximum utility
	if nSearch > 0 
		for ee = 1:size(u,2)
			v[3,ee] += exp(-smo*(u[r[ind_p],ee] - u[r[1],ee]))
			for nn in path
				v[3,ee] += exp(-smo*(u[r[ind_p],ee] - u[r[nn],ee]))
			end
			v[3,ee] -= 1 # adjustment since also includes purchased above
		end
	end
  	return
end


###################################################
function estimateModel(m::RandomSearch,d; startvals = [] , pars_true = []  )

	# Preliminaries 
	cfun = 	getCfun(m.cfun)
	
	nBeta = size(d.chars,2) 

	if isempty(pars_true)
		pars_true = fill(NaN,nBeta+1)
	end

	if isempty(startvals)
		startvals = rand(Uniform(),nBeta+1) 
	end
	
	# Random draws, pre-allocate 
	Random.seed!(m.seed)
	e = rand(m.dE,size(d.chars,1),m.options.nDraws)
	v = rand(m.dV,size(d.chars,1),m.options.nDraws)

	############################################################################
	# Otimization

	# Define objective function 
	obj_fun(θ,p) = negLogLik(m,θ,d,cfun;v=v,e=e)
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
	β = pars_opt[1:end-1]
	cs = pars_opt[end]
	finalVal = fval_opt

	infotxt = "
	####################################
	# Results of Estimation:
	Estimation model:	$(typeof(m))
	Final Objective Value: 		$finalVal
	Start Objective Value:		$startVal
	True Objective Value:		$trueVal
	Parameters:
		beta  = $(round.(β,digits=2))
		beta0 = $(round.(pars_true[1:end-1],digits=2))
		cs = $(round(cs,digits=2)) | cs0 = $(round(pars_true[end],digits=2))
	#################################### "

	@info infotxt

	return β,cs,finalVal

end


######################################################
function calcWelfareDemand(m::RandomSearch,d; e = [] , v = [] )

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
	maxProd = maximum(vcat([length(d.seli[i]) for i = 1:length(d.seli)]))

	# Form utility
	xβv = d.chars * m.β .+ v
	u   = xβv .+ e 

	# Search costs  
	cfun = getCfun(m.cfun)

	μ  = sum(mean(m.dChars) .* m.β[1:end-m.outDummy]) + mean(m.dV)
	σ2  = sum(var(m.dChars) .* m.β[1:end-m.outDummy] .^2) + var(m.dV)
	zs = calcZs(0.0,Normal(μ,sqrt(σ2)),cfun(m.cs))[1]

	# Pre allocate 
	sumU = zeros(Threads.nthreads())
	sumC = zeros(Threads.nthreads())
	demand = zeros(Int64,maxProd,Threads.nthreads())
	cs = cfun(m.cs) # Note, need to put here as cfun(m.cs) does not work in threaded
	@inbounds Threads.@threads for k = 1:Threads.nthreads()	
		threadRange = getrange(length(d.seli))

		locPathi 	= zeros(Int64,maxProd)
		locU = 0.0 
		locC = 0.0 
		locD = zeros(Int64,maxProd)
		for ii in threadRange
			@views posi = d.pos[ii]
			for dd in 1:size(e,2)
				@views ud 	= u[d.seli[ii],dd]
				locPathi 	.= 0 
				purchi 	= genSearchPathi!(m,locPathi,zs,ud)
				locU 	+= ud[purchi]
				for ss = 2:maxProd # NOTE: locPathi first entry is = 0 due to outside option
					if locPathi[ss] == 0 
						break
					else
						locC += cs # note: cs = cfun(m.cs) from above
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