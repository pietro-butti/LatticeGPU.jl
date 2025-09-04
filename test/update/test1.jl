###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    test1.jl
### created: Tue Sep  2 16:29:48 2025
###                               

using LatticeGPU, Test, CUDA

T = Float64
lp = SpaceParm{4}((4,4,4,4), (4,4,4,4), BC_PERIODIC, (0,0,0,0,0,1))
gp = GaugeParm{T}(SU3{T}, 6.5, 1.0)
ymws = YMworkspace(SU3, T, lp)

randomize!(ymws.mom, lp, ymws)
U = exp.(ymws.mom)
act = gauge_action(U, lp, gp, ymws)
plq = plaquette(U, lp, gp, ymws)
pl_exact = Eoft_plaq(U, gp, lp, ymws) 
cl_exact = Eoft_clover(U, gp, lp, ymws)
println("## Random config: ")
println("  - act: ", act)
println("  - plq: ", plq)
println("  - Epl: ", pl_exact)
println("  - Ecl: ", cl_exact)

updt_or_wilson!(U, gp, lp)

plq = plaquette(U, lp, gp, ymws)
act = gauge_action(U, lp, gp, ymws)
pl_exact = Eoft_plaq(U, gp, lp, ymws) 
cl_exact = Eoft_clover(U, gp, lp, ymws)
println("## After OR: ")
println("  - act: ", act)
println("  - plq: ", plq)
println("  - Epl: ", pl_exact)
println("  - Ecl: ", cl_exact)
