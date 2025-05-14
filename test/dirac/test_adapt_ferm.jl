using LatticeGPU, Test, CUDA

T = Float64
lp = SpaceParm{4}((16,16,16,16), (4,4,4,4), BC_PERIODIC, (0,0,0,0,0,0))
gp = GaugeParm{T}(SU3{T}, 6.1, 1.0)
dpar = DiracParam{T}(SU3fund,1.3,0.9,(1.0,1.0,1.0,1.0),0.0,0.0)
ymws = YMworkspace(SU3, T, lp)
dws = DiracWorkspace(SU3fund,T,lp);

randomize!(ymws.mom, lp, ymws)
U = exp.(ymws.mom)

psi = scalar_field(Spinor{4,SU3fund{T}},lp);
pfrandomize!(psi,lp)

Ucp = deepcopy(U)
psicp = deepcopy(psi)
# First Integrate very precisely up to t=2 (Wilson)
println(" # Very precise integration ")
wflw = wfl_rk3(Float64, 0.0004, 1.0E-7)
flw(U,psi, wflw, 5000, gp,dpar, lp, ymws, dws)
pl_exact = Eoft_plaq(U, gp, lp, ymws)
cl_exact = Eoft_clover(U, gp, lp, ymws)
println("  - Plaq:   ", pl_exact)
println("  - Clover: ", cl_exact)
Ufin = deepcopy(U)
psifin = deepcopy(psi)


# Now use Adaptive step size integrator:
for tol in (1.0E-4, 1.0E-5, 1.0E-6, 1.0E-7, 1.0E-8)
    local wflw = wfl_rk3(Float64, 0.0001, tol)
    U .= Ucp
    psi .= psicp
    ns, eps = flw_adapt(U,psi, wflw, 2.0, gp,dpar,lp, ymws,dws)
    pl = Eoft_plaq(U, gp, lp, ymws)
    cl = Eoft_clover(U, gp, lp, ymws)
    psierr = sum(norm2.((psi.-psifin)))./prod(lp.iL)

    println(" # Adaptive integrator (tol=$tol): ", ns, " steps")
    U .= U ./ Ufin
    maxd = CUDA.mapreduce(dev_one, max, U, init=0.0)
    println("  - Plaq:   ", pl," [diff: ", abs(pl-pl_exact), "; ",
            maxd, "]")
    println("  - Clover: ", cl, " [diff: ", abs(cl-cl_exact), "; ",
            maxd, "]")
    println("  - Fermion diff: ", psierr)
end
