using CUDA, LatticeGPU, TimerOutputs

#Check that Dw ( (DwdagDw)^{-1} g5 Dw g5 ) psi = psi for random fields

@timeit "Rand solver test" begin

@timeit "Generate random fields" begin

    lp = SpaceParm{4}((16,16,16,16), (4,4,4,4), 0, (0,0,0,0,0,0))
    gp = GaugeParm{Float64}(SU3{Float64}, 6.0, 1.0)
    ymws = YMworkspace(SU3, Float64, lp)
    dpar = DiracParam{Float64}(SU3fund,2.3,0.0,(1.0,1.0,1.0,1.0),0.0,0.0)
    dws = DiracWorkspace(SU3fund,Float64,lp);

    randomize!(ymws.mom, lp, ymws)
    U = exp.(ymws.mom)

    rpsi = scalar_field(Spinor{4,SU3fund{Float64}},lp)
    pfrandomize!(rpsi,lp)

end

prop = scalar_field(Spinor{4,SU3fund{Float64}},lp)

function krnlg5!(src)
    b=Int64(CUDA.threadIdx().x)
    r=Int64(CUDA.blockIdx().x)
    src[b,r] = dmul(Gamma{5},src[b,r])
    return nothing
end

CUDA.@sync begin
        CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnlg5!(rpsi)
end
g5Dw!(prop,U,rpsi,dpar,dws,lp)
CG!(prop,U,DwdagDw!,dpar,lp,dws,10000,1.0e-14)

Dw!(dws.sp,U,prop,dpar,dws,lp)

CUDA.@sync begin
    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnlg5!(rpsi)
end

res = sum(norm2.(rpsi-dws.sp))


if res < 1.0e-6 
    print("Drand test passed with ",res,"% error!\n")
    
else
    error("Drand test failed with difference: ",res,"\n")
end

end
