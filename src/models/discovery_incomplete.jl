function genData(m::SearchDiscovery,nCons::Int,nProdTot::Int,nProdCons::Int; 
					fname::String = "" , weights::AbstractWeights = fweights(fill(1,nProdTot-1)))
	
	Random.seed!(m.seed)
	# Initial checks
	@argcheck round((nProdCons-1-m.nA0)/m.nd)==(nProdCons-1-m.nA0)/m.nd  "Number of products (per consumer) does not match m.nA0 and m.nd combo."

	# Search cost specification  (relevant for Weitzman)
	cfun	= 	getCfun(m.cfun)
	zdfun   =   getZdfun(m.zdfun)

	# Consumer index into full data 
	seli = [ 1+ii*nProdCons:(ii+1)*nProdCons for ii = 0:nCons-1]

	# Products' positions (same for all consumers) 
	pos = fill([zeros(Int64,1+m.nA0) ; repeat(collect(Int64,1:(nProdCons-1-m.nA0)/m.nd),inner=m.nd)],nCons)

	# Generate products 

	pid,chars = genProducts(m::Model,seli,nProdTot,nProdCons,weights=weights)

	# Gather data in NamedTuple for convenience
	d = (seli = seli,pid = pid,chars = chars,pos = pos)

	# Discovery values (same for all)
	# NOTE: Distribution does not include outside option dummy
	nChar = length(m.dChars)
	μxb = mean(m.dChars)'*m.β[1:nChar] .+ mean(m.dV) 
	σxb = m.β[1:nChar]'*cov(m.dChars)*m.β[1:nChar] + var(m.dV)
	Gb = Normal(0,sqrt(σxb)) 
	Ξ = calcZd(Gb,cfun(m.cs),m.cd,m.nd) # calculates based on mean zero (see paper for details)
	zd = μxb .+ zdfun.(Ξ,m.dze,unique(pos[1])) 

	# Utility
	e   = rand(m.dE,size(d.chars,1))
	v   = rand(m.dV,size(d.chars,1))
	xβ  = d.chars * m.β
	u   = xβ .+ v .+ e 

	# Search values
	ξ = calcZs(0.0,m.dE,cfun(m.cs))[1]
	zs = xβ .+ v .+ ξ

	# Generate search paths 
	path,purch,stop,searched = genSearchPaths(m,d,nProdCons,zd,zs,u)
	
	# Gather again data 
	path = [path[path[:,i].>0,i] for i =1:nCons] # reduces path to array of arrays
	searched = [searched[:,i] for i = 1:nCons]
	d = (seli = seli,pid = pid,chars = chars,pos = pos,path = path, purch = purch, stop = stop, searched = searched, u=u , zs = zs, zd = zd)

	if !isempty(fname)
		@info "Saved data in bson format to $fname."
		@save fname d
	end

	return d
end

###################################################
function genSearchPaths(m::SearchDiscovery,d,maxProd,zd,zs,u)
	n = length(d.seli)

	# Same for all consumers 
	path = zeros(Int64,maxProd,n)
	purch = fill(0,n) 
	stop  = fill(0,n) 
	searched = fill(false,maxProd,n)


  @inbounds Threads.@threads for k = 1:Threads.nthreads()	
		threadRange = getrange(n)
		locPathi 	= zeros(Int64,maxProd)
		locSearchi  = fill(false,maxProd)
		locO 		= fill(false,maxProd,2)

		for ii in threadRange # Loop over consumers allocated to specific thread 
			# Select consumer 
			r = d.seli[ii]
			nProd = length(d.seli[ii])
			# Reset locals 
			locPathi .= 0
			locSearchi .= false 
			locO .= false
			locO[1,2] = true
			if m.nA0 > 0 
				locO[2:m.nA0+1,1] .= true # Initial awareness set 
			end
			# generate search paths (updates locPathi)
			purchi,stopi = genSearchPathi!(m,locPathi,locSearchi,zs[r],u[r],locO,zd,d.pos[ii])
			# Fill in values 
			path[:,ii] = locPathi 
			purch[ii] = purchi
			stop[ii] = stopi
			searched[:,ii] = locSearchi
		end

	end

	return path,purch,stop,searched
end

###################################################
function genSearchPathi!(m::SearchDiscovery,path,searched,zs,u,O,zd,pos)
	
	# Initial status 
	cc_pos = 0  # position counter
	ind_pos = 1+m.nA0+1 # position index 
	cc_s = 1 # counter of how many searches
	ind_p = 1 # index of maximum utility (initially outside option)
	u_max = u[ind_p] # max utility
	maxPos = maximum(pos)
	# Generate search paths 
	while true
		# Discover more (unless reached end)
		val_s,ind_s = findmax_subsel(zs,O[:,1])
		if cc_pos < maxPos && zd[cc_pos+1] >  max(val_s,u_max) 
			cc_pos += 1
			O[ind_pos:ind_pos+m.nd-1,1] .= true # put in awareness set 
			ind_pos += m.nd
		# Purchase and end search
		elseif u_max >= val_s
			# Note that only reaches this condition if zd < y
			# -Inf added since S otherwise might be empty
			purch = ind_p # d[i].pid[ind_p]
			# path = path[1:cc_s-1]
			stop  = cc_pos 
			return purch,stop
			break # end with purchase
		# Search best from awareness set 
		else 
			# Update awareness and consideration set 
			O[ind_s,1] = false
			O[ind_s,2] = true
			path[cc_s] = ind_s 
			searched[ind_s] = true 
			cc_s += 1
			if u[ind_s] >= u_max
				ind_p = ind_s
				u_max = u[ind_p]
			end
		end
	end
end

######################################################
function calcWelfareDemand(m::SearchDiscovery,d; e = [] , v = [] )

	# Draw random numbers if not provided 
	if isempty(e)
		Random.seed!(m.seed+4123)
		e = rand(m.dE,size(d.chars,1),m.options.nDraws)
	end
	if isempty(v)
		Random.seed!(m.seed+2333)
		v = rand(m.dV,size(d.chars,1),m.options.nDraws)
	end
	
	# Derived values 
	nCons = length(d.seli)
	maxProd = maximum(vcat([length(d.seli[i]) for i = 1:length(d.seli)]))

	# Form utility
	xβv = d.chars * m.β .+ v
	u   = xβv .+ e 

	# Search cost specification  
	cfun	= 	getCfun(m.cfun)
	zdfun   =   getZdfun(m.zdfun)
	cs = cfun(m.cs) # Note, need this here as cfun(m.cs) does not work in threaded
	
	# Search values (same for all consumers)
	ξ   = calcZs(0.0,m.dE,cs)[1]
	zs  = xβv .+ ξ

	# Discovery values 
	μx = mean(m.dChars)' * m.β[1:end-m.outDummy] .+ mean(m.dV)
	σx = sum(m.β[1:end-m.outDummy].^2 .* var(m.dChars)) .+ var(m.dV)
	G   = Normal(0, sqrt(σx))
	zd0 = calcZd(G,cs,m.cd,m.nd)
	zd  = μx .+ zdfun.(zd0,m.dze,0:length(d.seli[1]))

	# Pre allocate 
	sumU = zeros(Threads.nthreads())
	sumC = zeros(Threads.nthreads())
	demand = zeros(Int64,maxProd,Threads.nthreads())
	@inbounds Threads.@threads for k = 1:Threads.nthreads()	
		threadRange = getrange(length(d.seli))

		locPathi 	= zeros(Int64,maxProd)
		locSearchi  = fill(false,maxProd)
		locO 		= fill(false,maxProd,2)
		
		locU = 0.0 
		locC = 0.0 
		locD = zeros(Int64,maxProd)
		for ii in threadRange
			@views posi = d.pos[ii]
			for dd in 1:size(e,2)
				@views zsd = zs[d.seli[ii],dd]
				@views ud = u[d.seli[ii],dd]
				locPathi .= 0 
				locO .= false 
				locO[1,2] = true
				if m.nA0 > 0 
					locO[2:m.nA0+1,1] .= true # Initial awareness set 
				end	
				purchi,stopi = genSearchPathi!(m,locPathi,locSearchi,
								zsd,ud,locO,zd,posi)
				locU += ud[purchi]
				locC += cs * (findfirst(isequal.(locPathi,0)) - 1) + m.cd * stopi
				locD[purchi] += 1 
			end
		end
		sumU[k] = locU / size(e,2)
		sumC[k] = locC / size(e,2)
		demand[:,k] .= locD 
	end
	return sum(sumU) - sum(sumC), sum(sumU), sum(sumC), sum(demand,dims=2) ./ size(e,2) ./ nCons
end