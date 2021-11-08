################################################################################
# Function that writes tabular data into table
# Latex dependencies: 
# - \usepackage[flushleft]{threeparttable}
# - \usepacakge{booktabs}
# Needs threeparttable package (\usepackage[flushleft]{threeparttable}) in Latex
function writeLatexTable(d,fname,caption,colnames,rownames,comment;
	label = "", midrules=[],format="%.2f", pos = "htb",percentcols = [])
	f = open(fname,"w+")

	n = size(d,1)
	k = size(d,2)

	tt = ["\\begin{table}[$pos] \\centering ";
			"\\caption{$caption} \\label{$label}" ;
			"\\begin{threeparttable}" ;
			"\\begin{tabular}{$(repeat("c",k+2))} ";
			"\\midrule"]
	for ii = 1: size(tt,1)
		println(f,tt[ii])
	end
	print(f,"&")
	for ii = 1:k
		print(f,"& $(colnames[ii])")
	end
	println(f,"\\\\")
	println(f,"\\midrule")
	for ii = 1:n
		print(f, rownames[ii])
		print(f,"&")
		for jj = 1:k
			print(f,"&")
			if !isnan(d[ii,jj])
				fmttxt = @eval @sprintf $format $d[$ii,$jj]
				if jj in percentcols 
					fmttxt *= "\\%"
				end
				print(f,fmttxt)
			else
				print(f," ")
			end
		end
		println(f,"\\\\")
		if !isempty(midrules) && midrules[ii]==1
			println(f,"\\midrule")
		end

	end
	println(f,"\\midrule ")
	println(f,"\\end{tabular}")
	if !(isempty(comment))
		println(f,"\\begin{tablenotes}")
		println(f,"\\item \\footnotesize{\\emph{Notes:} $comment }")
		println(f,"\\end{tablenotes} ")
	end
	println(f,"\\end{threeparttable}")
	println(f,"\\end{table}")

	close(f)
end

#####
# Overload functions in Distributions 
import Distributions: cov
# Required since cov(::UnivariateDistribution) not defined in Distributions.jl
function cov(d::UnivariateDistribution)
	return var(d)
end


#####################################################################
# Printing

function printD(d,i)
	println("Purch = $(d[i].purch) ")
	println("Stop = $(d[i].stop) ")
	println("Zd = $(round.(d[i].zd,digits=2))")
	println("Pid | Pos  | Path | Zs   | u ")
	display(round.([d[i].pid d[i].pos d[i].path d[i].zs d[i].u ],digits=2))
end


#####################################################################
# Max/min functions used in code (for performance) 
@inline function findmax_subsel(A,S)
	val, ind = typemin(eltype(A)), -1
	for ii in eachindex(A)
		val_ = A[ii]
		if val_ > val && S[ii] == true
			val, ind = val_, ii
		end
	end
	return val, ind
end

@inline function findmax_range(A,R)
	val, ind = typemin(eltype(A)), -1
	for ii in R
		val_ = A[ii]
		if val_ > val
			val, ind = val_, ii
		end
	end
	return val, ind
end

@inline function findmin_range(A,R)
	val, ind = typemax(eltype(A)), -1
	for ii in R
		val_ = A[ii]
		if val_ < val
			val, ind = val_, ii
		end
	end
	return val, ind
end


@inline function findmax_range(A,R,ee)
	val, ind = typemin(eltype(A)), -1
	for ii in R
		val_ = A[ii,ee]
		if val_ > val
			val, ind = val_, ii
		end
	end
	return val, ind
end

#####################################################################
# Get range within thread 
@inline function getrange(n)
	tid = Threads.threadid()
	nt = Threads.nthreads()
	d , r = divrem(n, nt)
	from = (tid - 1) * d + min(r, tid - 1) + 1
	to = from + d - 1 + (tid ≤ r ? 1 : 0)
	from:to
end

#####################################################################
# Translate string cost spec into a function 

function getCfun(s::String)
	if s == "linear" 
		(cs::Float64) -> cs
	elseif s == "exp" 
		(cs::Float64) -> exp(cs)
	elseif s == "quadratic"
		(cs::Float64) -> cs^2
	else
		error("cfun not correctly specified")
	end
end

function getZdfun(s::String)
	if s == "linear"
		(z,dze,pos) -> z + dze * pos 
	elseif s.zdfun == "exp"
		(z,dze,pos) -> z - exp(dze*pos) + 1.0
	elseif s.zdfun == "quadratic"
		(z,dze,pos) -> z + dze * pos^2 
	else
		error("zdfun not correctly specified")
	end
end


#####################################################################
# Draw products
function genProducts(m::Model,seli,nProdTot,nProdCons; 
						weights::AbstractWeights= fweights(fill(1,nProdTot-1)))
		# Derived values
		nChar = length(m.dChars) # number of characteristics (excl. outside option dummy)
		nCons = length(seli)
		# Product ids, 0 will be outside option 
		pid = 1:nProdTot-1 ; 
		
		# Product characteristics 
		if nChar == 1 
			chars = [0.0 ;rand(m.dChars,nProdTot-1,1)]
		else
			chars = [fill(0.0,1,nChar) ; permutedims(rand(m.dChars,nProdTot-1),[2,1])]
		end
	
		# Empty Array for consumer-products 
		charsC = zeros(seli[end][end],nChar + m.outDummy)

		# Draw products per consumer randomly from set of products for each consumer 
		pidC = [ [0 ; sample(pid,weights,nProdCons-1,replace=false)] for i = 1:nCons]

		# Fill into array for consumer-products 
		for ii = 1:nCons
			charsC[seli[ii],1:end-m.outDummy] = chars[pidC[ii] .+ 1,:]
			if m.outDummy 
				charsC[seli[ii][1],end] = 1.0 
			end
		end
	return pidC,charsC
end
