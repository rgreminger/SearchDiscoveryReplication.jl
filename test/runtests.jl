using SearchDiscoveryReplication
using Test

# Required packages
using Distributions,Random

# Test sets
@testset "Reservation values" begin

	# Search value
	F = Distributions.DiscreteNonParametric([-1/2,1/2],[1/2,1/2])
	@test round(calcZs(0,F,0.1)[1] ,digits=5)== 0.3

	F = Distributions.Normal() ;
	@test round(calcZs(0,F,0.1)[1] ,digits=5)== 0.90235
	@test round.(calcZs([0,1],F,0.1) ,digits=5) == [0.90235,1.90235]

	# Expansion value
	nd = 1 ; cs = 0.1 ; cd = 0.1
	G = Distributions.Normal()
	@test round(calcZd(G,cs,cd,nd),digits=5) == 1.20125

	G = Distributions.Normal(0.2,0.4)
	@test round(calcZd(G,cs,cd,nd),digits=5) == 0.80748

	nd = 2 ; G = Distributions.Normal()
	@test round(calcZd(G,cs,cd,nd),digits=5) == 1.58767

end


@testset "Weitzman model" begin

	m = Weitzman( 
		β = [2.0,-0.4], cs = 0.2 , cd = 0.4 ,
		nd = 1, nA0 = 2 , dChars =  MvNormal([1,1],1/2), outDummy=false,
		seed = 234,dV= Normal(0,0), dE = Normal(),cfun = "linear", smo = 25,
		options=estimOptions(nDraws =3)
		)
	nCons = 5
	nProdTot = 10
	nProdCons = 3 
	d = genData(m,nCons,nProdTot,nProdCons )
	@test d.path[1] == [3]
	@test d.purch[1] == 3			  
	@test d.search[1:3] == [1,0,1] 

	cfun = SearchDiscoveryReplication.getCfun(m.cfun)
	Random.seed!(m.seed)
	maxPos = maximum(d.pos)

	@test SearchDiscoveryReplication.negLogLik(m,vcat(m.β,m.cs,m.cd),length(m.β),d,cfun,maxPos) ≈ 14.5350 atol = 1e-4 

	wd = calcWelfareDemand(m,d) 
	@test wd[1]		 	≈ 11.8747 atol = 1e-4
	@test wd[2]	 		≈ 12.9414 atol = 1e-4
	@test wd[4][1]	  	≈ 0.1333 atol = 1e-4
	@test wd[4][end]	≈ 0.5333 atol = 1e-4
end


@testset "FullInfo model" begin
	m = FullInfo(β = [2.0,-0.4],dChars = MvNormal([1,1],1/2), outDummy = false, seed = 234, 
					dV= Normal(0,0), dE = Normal(),smo=25,options=estimOptions(nDraws =3))
	nCons = 5
	nProdTot = 10
	nProdCons = 3 
	d = genData(m,nCons,nProdTot,nProdCons )
	@test d.purch[1] == 3			  
	Random.seed!(m.seed)
	@test SearchDiscoveryReplication.negLogLik(m,m.β,d) ≈ 42.8430 atol = 1e-4 

	wd = calcWelfareDemand(m,d) 
	@test wd[1]		 	≈ 12.9784 atol = 1e-4
	@test wd[4][1]		≈ 0.0666 atol = 1e-4
	@test wd[4][end]	≈ 0.4 atol = 1e-4
end

@testset "RandomSearch model" begin
	m = RandomSearch(β = [2.0,-0.4],cs = 0.3,dChars = MvNormal([1,1],1/2), outDummy = false,
			seed = 234, dV= Normal(0,0), dE = Normal(),cfun = "linear",
			smo=5,options=estimOptions(nDraws =3))
	nCons = 5
	nProdTot = 10
	nProdCons = 3 
	d = genData(m,nCons,nProdTot,nProdCons )
	@test d.purch[1] == 3			  
	@test isequal(d.path[1],[2,3])
	Random.seed!(m.seed)
	cfun = SearchDiscoveryReplication.getCfun(m.cfun)
	@test SearchDiscoveryReplication.negLogLik(m,vcat(m. β,m.cs),d,cfun) ≈ 11.1138 atol = 1e-4 

	wd = calcWelfareDemand(m,d) 
	@test wd[1]			≈ 10.2291 atol = 1e-4
	@test wd[2]		 	≈ 12.2291  atol = 1e-4
	@test wd[4][1]		≈ 0.1333 atol = 1e-4
	@test wd[4][end]	≈ 0.2666 atol = 1e-4
end

@testset "SearchDiscovery model" begin
	m = SearchDiscovery(β = [1.2,0.4],cs = 0.3,cd = 0.3,dze = -0.1, dChars = MvNormal([0.4,0.5],[0.8,1.2]),
		outDummy = false, seed = 23, nd = 1 , nA0 = 1, dV= Normal(0,1), dE = Normal(),cfun = "linear", zdfun = "linear",
		options=estimOptions(nDraws =200))

	nCons = 30
	nProdTot = 10
	nProdCons = 5  
	d = genData(m,nCons,nProdTot,nProdCons)
	@test isequal(d.path[1],[2,3])
	@test d.purch[1] == 3
	@test d.stop[1] == 1

	wd = calcWelfareDemand(m,d) 
	@test wd[1]			≈ 45.5269 atol = 1e-4
	@test wd[2]			≈ 65.0179 atol = 1e-4
	@test wd[4][1]		≈ 0.1866 atol = 1e-4
	@test wd[4][end]	≈ 0.1195 atol = 1e-4
end