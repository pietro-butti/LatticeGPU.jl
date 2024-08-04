using CUDA

using Pkg

Pkg.activate("/home/fperez/Git/LGPU_fork_ferflow")

using LatticeGPU

lp = SpaceParm{4}((4,4,4,4),(2,2,2,2),0,(0,0,0,0,0,0));

pso = scalar_field(Spinor{4,SU3fund{Float64}},lp);
psi = scalar_field(Spinor{4,SU3fund{Float64}},lp);
psi2 = scalar_field(Spinor{4,SU3fund{Float64}},lp);

ymws = YMworkspace(SU3,Float64,lp);
dws = DiracWorkspace(SU3fund,Float64,lp);

int = wfl_rk3(Float64, 0.01, 1.0)

gp = GaugeParm{Float64}(SU3{Float64},6.0,1.0,(1.0,0.0),(0.0,0.0),lp.iL)

dpar = DiracParam{Float64}(SU3fund,1.3,0.9,(1.0,1.0,1.0,1.0),0.0)

randomize!(ymws.mom, lp, ymws)
U = exp.(ymws.mom);

pfrandomize!(psi,lp)
for L in 4:19
    pso .= psi
    V = Array(U)
    a,b = flw_adapt(U, psi, int, L*int.eps, gp,dpar, lp, ymws,dws)
     # for i in 1:a
     # flw(U, psi, int, 1 ,b[i], gp, dpar, lp, ymws, dws)
     # end
    pfrandomize!(psi2,lp)

    foo = sum(dot.(psi,psi2))# field_dot(psi,psi2,sumf,lp)
    copyto!(U,V);
    backflow(psi2,U,L*int.eps,7,gp,dpar,lp, ymws,dws)
    println("Error:",(sum(dot.(pso,psi2))-foo)/foo)
    psi .= pso
end
