###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    test_adapt.jl
### created: Mon Jun  6 12:01:36 2022
###                               

using LatticeGPU, Test, CUDA

T = Float64
lp = SpaceParm{4}((16,16,16,16), (4,4,4,4), BC_PERIODIC, (0,0,0,0,0,0))
gp = GaugeParm{T}(SU3{T}, 6.1, 1.0)
ymws = YMworkspace(SU3, T, lp)

randomize!(ymws.mom, lp, ymws)
U = exp.(ymws.mom)

Ucp = deepcopy(U)
# First Integrate very precisely up to t=2 (Wilson)
println(" # Very precise integration ")
wflw = wfl_rk3(Float64, 0.0004, 1.0E-7)
flw(U, wflw, 5000, gp, lp, ymws)
pl_exact = Eoft_plaq(U, gp, lp, ymws) 
cl_exact = Eoft_clover(U, gp, lp, ymws)
println("  - Plaq:   ", pl_exact)
println("  - Clover: ", cl_exact)
Ufin = deepcopy(U)


# Now use Adaptive step size integrator:
for tol in (1.0E-4, 1.0E-5, 1.0E-6, 1.0E-7, 1.0E-8)
    local wflw = wfl_rk3(Float64, 0.0001, tol)
    U .= Ucp
    ns, eps = flw_adapt(U, wflw, 2.0, gp, lp, ymws)
    pl = Eoft_plaq(U, gp, lp, ymws) 
    cl = Eoft_clover(U, gp, lp, ymws)

    println(" # Adaptive integrator (tol=$tol): ", ns, " steps")
    U .= U ./ Ufin
    maxd = CUDA.mapreduce(dev_one, max, U, init=0.0)
    println("  - Plaq:   ", pl," [diff: ", abs(pl-pl_exact), "; ",
            maxd, "]")
    println("  - Clover: ", cl, " [diff: ", abs(cl-cl_exact), "; ",
            maxd, "]")
end
