using Distributions, SearchDiscoveryReplication, NLopt, GalacticOptim
using BSON
using PDMats # required for BSON to load data again 

##############################################
## Run simulation for main tables 

# Set parameter values  
β0 = [1.0 , -1.0,3.5]
cs0 = 0.03
cd0 = 0.06
dChars = MvNormal([2., 3.5], [3.0, 1.0])
outDummy = true
dV = Normal(0, 0)
nCons = 2000
# By having many products in total, it is unlikely that the same product is shown to two  
# different consumers
nProdTot = 1000000 
nProdCons = 30 
nA0 = 1

# Run simulation 
d, wSD, outFI, outRS, outWf, outW = runSimulation(β0, cs0, cd0, dChars, outDummy, dV, nCons, nProdTot, 
										nProdCons,nA0; 
										seed =24,
										options = estimOptions(algo = Opt(:LN_BOBYQA,1))) 

# Save results as bson 

BSON.@save "gen/results_spec0.bson" wSD outFI outRS outWf outW β0 cs0 cd0 dChars outDummy dV nA0 
# BSON.@load "gen/results_spec0.bson" wSD outFI outRS outWf outW β0 cs0 cd0 dChars outDummy dV nA0 

############################################## 
## Gather results  and print to tables
estimOut, welfareChangesOut, demandOut, dStats = gatherResults("gen/results_spec0.bson")  


# TABLE 2: Estimation Results 
colnames = 	["\$ \\beta_2 \$", "\$ \\beta_1 / | \\beta_2 |\$",
				"\$ \\beta_3 / | \\beta_2 | \$ ", 
				"\$ c_s / | \\beta_2 | \$", "\$ c_d / | \\beta_2 |\$"]
rownames = ["\\emph{SD}","\\emph{DS1}","\\emph{DS2}","\\emph{RS}","\\emph{FI}"]
td = deepcopy(estimOut')[:,1:end-1]

# Normalize with price coefficient 
for i in [1,3,4,5]
	global td[:,i] = td[:,i] ./ abs.(td[:,2])
end
td = td[:,[2,1,3,4,5]] 

# nSearches and share outside Option 
colnames = prepend!(colnames,["\\#Searches","Purchases (\\%)"])
dStats[:,2] .= (1 .- dStats[:,2]) .* 100  # Make into % share purchases 

td = hcat(dStats,td) 

title =  "Estimated Coefficients and Search Set Size"
notes = "Estimation from a simulated dataset with 2,000 consumers and 30 products per consumer. 
Characteristics are independent draws (across consumers and products) from 
\$x_{1j} \\sim N(2,3.0)\$, \$x_{2j} \\sim N(3.5,1.0)\$ and \$y_j \\sim N(0,1)\$, , with parameters in the 
estimated models denoted by \$c^{RS}=c_s\$, \$c_j^{DS1} = c_s\$ and \$c_J^{DS2} = c_s + c_d h_j\$. The third characteristic is an outside dummy.
The data is generated based on the \\emph{SD} model with \$ n_d=|A_0|=1\$. The first two columns are based either 
on the generated data (SD) or estimated by generating 5,000 search paths for each consumer. "

writeLatexTable(td,"./gen/table2_coefs.tex",title,
				colnames , rownames, notes; 
				label = "tab:coef_estim")

# TABLE 3: Counterfactuals 
colnames = repeat(["\$\\Delta CS \$", "\$\\Delta D_1\$","\$\\Delta D_5 \$"],2)

td = zeros(size(welfareChangesOut[1],1),1)
for i in 1:2 
	global td = hcat(td,welfareChangesOut[1][:,1+i],demandOut[1][:,i],demandOut[2][:,i])
end
td = td[:,2:end]

title = "Counterfactuals"
notes = "Results from simulated counterfactuals based on Table \\ref{tab:coef_estim}, where (i) all costs are set to zero and (ii) the price for the 5th 
is reduced by 1 \\% for each consumer. All changes are expressed in \\% relative to the baseline. Demand and consumer
surplus are calculated by averaging across 5,000 simulated search paths for each consumer."

writeLatexTable(td,"./gen/table3_counterfactuals.tex",title,
				colnames , rownames, notes; 
				label = "tab:counterfactuals" ) 

# Insert table headers 
flines = open(readlines,"gen/table3_counterfactuals.tex")
addedLines = ["&& \\multicolumn{3}{c}{Remove costs} & \\multicolumn{3}{c}{\$\\Delta p_5 = -1\\%\$} \\\\ ",
				"\\cline{3-5} \\cline{6-8}"]

flines = vcat(flines[1:5],addedLines,flines[6:end])
f = open("gen/table3_counterfactuals.tex","w")
for l in flines 
	println(f,l)
end
close(f)


				