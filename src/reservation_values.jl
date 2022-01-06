################################################################################
# Discovery value
"""
	calcZd(G::Normal,cs,cd,nd) 

Returns discovery value. Calculation is shown in appendix of paper. For `nd=1`, a faster implementation based on bivariate normal distribution is used.

# Arguments:
- `G`: Distribution of valuations revealed on overview page. Must be Normal distribution. Assumes F is standard Normal. 
- `cs`: search costs
- `cd`: discovery costs
- `nd`: # products revealed per discovery
"""
function calcZd(G::Normal,cs,cd,nd)
	cs = max(cs,0)
	# Get ξ(cs) here instead of in EV(), as requires calculation only once
	ξ = min(calcZs(0,Normal(),cs)[1],1e6)  # limit, as with too large does not converge

	zd = if nd == 1 # Special case where more efficient calculation available
			if integrateCdfSingle(cd,ξ,cs,mean(G),std(G))-cd ≈ - cd  # case where no convergence 
				-cd 
			elseif cd <= 0 || (std(G) > 1e9 && cd <= 1e8) 
				Inf
			else
				fzero(t -> integrateCdfSingle(t,ξ,cs,mean(G),std(G))-cd,cd)
			end
		else
			if integrateCdfMax(cs,ξ,mean(G),std(G),nd)-cs-cd ≈ -cd 
				-cd 
			elseif cd <= 0 || (std(G) > 1e9 && cd < 1e8) 
				Inf
			else
				fzero(t -> integrateCdfMax(t,ξ,mean(G),std(G),nd)-t-cd,cd)
			end
		end
	return zd::Float64
end


function cdfW(z,ξ,μ,σ)
	a = 1/sqrt(1+σ^2)
	return cdf(Normal(),(z-ξ-μ)/σ) + cdf(Normal(),(z-μ)*a) - cdf(BiNormal(σ*a),[(z-μ)*a,(z-ξ-μ)/σ])
end

function integrateCdfMax(z,ξ,μ,σ,nd)
	return quadgk(t -> 1 - cdfW(t,ξ,μ,σ)^nd,z,Inf)[1] + z
end

function integrateCdfSingle(z,ξ,cs,μ,σ)
	a = z-μ ; b = z-ξ-μ ; c = 1/sqrt(1+σ^2) 

	f(x) = pdf(Normal(),x)
	F(x) = cdf(Normal(),x)

	return (1-F(b/σ))*(μ-z -cs)+σ*f(b/σ) +
			(z-μ) * (F(a*c) -cdf(BiNormal(σ*c),[a*c,b/σ]) )	- 
			σ*(-σ*c*f(a*c)*(1-F(b/σ/c-a*σ*c)) + F(a-b)*f(b/σ)) +
				c*f(a*c)*(1-F(b/σ/c-a*σ*c))
end

################################################################################
# Search value
"""
	calcZs(x,F::Distribution,cs::T) where T
	calcZs(x::Array{T,1},F::Distribution,cs::Vector ) where T

Returns search value(s)  zs = x .+ ξ(cs), where ξ(cs) is defined
as the lower bound of the integral in cs = ∫(1-F(ϵ))dϵ.

If both x and cs are 1d-arrays, then returns 1d-array zs, where each element z[i] is calculated based on x[i],cs[i]

# Arguments:
- `x`: partial valuation (in u = x + ϵ)
- `F`: distribution of ϵ
- `cs`: inspection costs
"""
function calcZs(x,F::Distribution,cs::T) where T
	zs = Array{T,1}(undef,length(x))
	if cs <= 0 || (std(F) > 1e9 && cs < 10)
		zs .= Inf
		return zs
	else
		if F == Normal() # Faster way when having std normal, and also structured to be suitable for Autodiff
			fz_N(cs) = fzero(ξ-> -ξ + ξ*cdf(F,ξ)+pdf(F,ξ)-cs ,-abs(cs)*10,100*std(F))
			ξ = fz_N(cs) 
			zs .= x .+ ξ
			return zs
		else
			fz(cs) = fzero(ξ->zs_inner_integral(ξ,F)-cs,-cs,30*std(F))
			ξ = fz(cs)
			zs .= x .+ ξ
			return zs
		end
	end
end

function calcZs(x::Array{T,1},F::Distribution,cs::Vector ) where T
	@argcheck size(x)==size(cs) "x and search costs need to have same size"
	zs = Array{T,1}(undef,length(x))
	@inbounds for cii in unique(cs)
		ξ = calcZs(0.0,F,cii)
		ind = (cs.==cii)
		zs[ind] .= x[ind] .+ ξ
	end
	return zs
end

"""
	zs_inner_integral(ξ,F)
	zs_inner_integral(ξ,F::Normal)
Returns ∫_ξ (1-F(ϵ))dϵ. Analytical expression for normal distribution available. 
"""
function zs_inner_integral(ξ,F)
	quadgk(e->(1-cdf(F,e)),ξ,maximum(F))[1]
end

function zs_inner_integral(ξ,F::Normal)
	 σ = std(F)
	 μ = mean(F)
	 return (1.0-cdf(Normal(),(ξ-μ)/σ))*(μ - ξ) + σ * pdf(Normal(),(ξ-μ)/σ)
 end

 