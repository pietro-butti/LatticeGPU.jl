import Pkg
Pkg.activate("/users/buttipie/LatticeGPU.jl")

using LatticeGPU



function test_basics()
    lp = SpaceParm{4}((16,16,16,16),(4,4,4,4),BC_PERIODIC,(0,0,0,0,0,0))
    intsch = omf4(Float64,0.01,50)

    phi = scalar_field(Float64,lp);
    U = vector_field(SU3{Float64},lp);

    println("Scalar and Gauge correctly allocated thorugh $(BACKEND[]) backend")
end



# Test with CPU backend
println("Testing CPU backend:")
set_backend!(CPUBackend())
test_basics()

# Test with AMDGPU backend (make sure AMDGPU.jl is installed)
println("\nTesting AMDGPU backend:")
set_backend!(AMDGPUBackend())
test_basics()



# fill!(U,one(SU3{Float64}));
# fill!(psi,zero(Float64));