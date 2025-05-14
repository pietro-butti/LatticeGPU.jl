using CUDA, LatticeGPU

println(" # Consistency condition for backflow")

lp = SpaceParm{4}((16,16,16,16), (4,4,4,4), BC_PERIODIC, (0,0,0,0,0,0))
pso = scalar_field(Spinor{4,SU3fund{Float64}},lp);
psi = scalar_field(Spinor{4,SU3fund{Float64}},lp);
psi2 = scalar_field(Spinor{4,SU3fund{Float64}},lp);

ymws = YMworkspace(SU3,Float64,lp);
dws = DiracWorkspace(SU3fund,Float64,lp);

int = wfl_rk3(Float64, 0.01, 1.0)

gp = GaugeParm{Float64}(SU3{Float64},6.0,1.0,(1.0,0.0),(0.0,0.0),lp.iL)

dpar = DiracParam{Float64}(SU3fund,1.3,0.9,(1.0,1.0,1.0,1.0),0.0,0.0)

randomize!(ymws.mom, lp, ymws)
U = exp.(ymws.mom);

pfrandomize!(psi,lp)
for L in 10:20:210
    pso .= psi
    V = Array(U)
    #a,b = flw_adapt(U, psi, int, L*int.eps, gp,dpar, lp, ymws,dws)
    flw(U, psi, int, L,int.eps, gp,dpar, lp, ymws,dws)
     # for i in 1:a
     # flw(U, psi, int, 1 ,b[i], gp, dpar, lp, ymws, dws)
     # end
    pfrandomize!(psi2,lp)

    foo = sum(dot.(psi,psi2))
    copyto!(U,V);
    backflow(psi2,U,L*int.eps,20,gp,dpar,lp, ymws,dws)
    println("# Consistency backflow test for t=",L*int.eps)
    println("Relative error:",abs((sum(dot.(pso,psi2))-foo)/foo))
    psi .= pso
end

