import Pkg
Pkg.activate("/users/buttipie/LatticeGPU.jl")

using LatticeGPU
using CUDA, AMDGPU


function krnl!(frc, m, lp::SpaceParm{N,M,BC_PERIODIC,D}) where {N,M,D}
    @inbounds begin
        # backend-specific part
        b = AMDGPU.Device.threadIdx().x |> Int64
        r = AMDGPU.Device.blockIdx().x |> Int64

        # common body part
        for id in 1:lp.ndim
            frc[b,id,r] = SU2alg(m[b,id,1,r], m[b,id,2,r], m[b,id,3,r])
        end
    end

    return nothing
end


function test_basics()
    lp = SpaceParm{4}((4,4,4,4),(2,2,2,2),BC_PERIODIC,(0,0,0,0,0,0))
    intsch = omf4(Float64,0.01,50)

    gp = GaugeParm{Float64}(SU2{Float64},2.45,5/3)
    ymws = YMworkspace(SU2,Float64,lp);


    bin = backend_randn(ymws.PRC, lp.bsz,lp.ndim,3,lp.rsz)
    begin
        AMDGPU.@roc groupsize=lp.bsz gridsize=lp.rsz krnl!(ymws.mom,bin,lp) 
        AMDGPU.synchronize()
    end

    println("=============== Ok ================")

    randomize!(ymws.mom, lp, ymws)
    println("=============== Ok ================")
end



# # Test with CPU backend
# println("Testing CPU backend:")
# set_backend!(CPUBackend())
# test_basics()

# Test with AMDGPU backend (make sure AMDGPU.jl is installed)
println("\nTesting AMDGPU backend:")
set_backend!(AMDGPUBackend())
test_basics()



