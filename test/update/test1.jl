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

using LatticeGPU, Test, CUDA, TimerOutputs

T = Float64
#lp = SpaceParm{4}((4,4,4,4), (4,4,4,4), BC_SF_ORBI, (0,0,0,0,0,0))
lp = SpaceParm{4}((16,16,16,16), (4,4,4,4), BC_PERIODIC, (0,0,0,0,0,1))
gp = GaugeParm{T}(SU3{T}, 6.5, 1.0, (1.5,1.5), (0.0,0.0), lp.iL[1])
ymws = YMworkspace(SU3, T, lp)

println(gp.Ubnd)
randomize!(ymws.mom, lp, ymws)
U = exp.(ymws.mom)
act = gauge_action(U, lp, gp, ymws)
plq = plaquette(U, lp, gp, ymws)
pl_exact = Eoft_plaq(U, gp, lp, ymws) 
cl_exact = Eoft_clover(U, gp, lp, ymws)
println(lp)
println("## Random config: ")
println("  - act: ", act)
println("  - plq: ", plq)
println("  - Epl: ", pl_exact)
println("  - Ecl: ", cl_exact)

Ucp = copy(U)

eo = evenodd(lp)
for i in 1:2000
    updt_or_wilson!(U, gp, eo, lp)
end
    
plq = plaquette(U, lp, gp, ymws)
act = gauge_action(U, lp, gp, ymws)
pl_exact = Eoft_plaq(U, gp, lp, ymws) 
cl_exact = Eoft_clover(U, gp, lp, ymws)
println("## After OR [eo]: ")
println("  - act: ", act)
println("  - plq: ", plq)
println("  - Epl: ", pl_exact)
println("  - Ecl: ", cl_exact)

U .= Ucp
for i in 1:2000
    updt_or_wilson!(U, gp, lp)
end

plq = plaquette(U, lp, gp, ymws)
act = gauge_action(U, lp, gp, ymws)
pl_exact = Eoft_plaq(U, gp, lp, ymws) 
cl_exact = Eoft_clover(U, gp, lp, ymws)
println("## After OR: ")
println("  - act: ", act)
println("  - plq: ", plq)
println("  - Epl: ", pl_exact)
println("  - Ecl: ", cl_exact)

print_timer(linechars = :ascii)
