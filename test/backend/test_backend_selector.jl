import Pkg
Pkg.activate("/users/buttipie/LatticeGPU.jl")

using LatticeGPU

# Function to print sin and cos of pi/4 using current backend
function test_sincos()
    x = π/4
    s = backend_sin(x)
    c = backend_cos(x)
    println("Testing math: sin(π/4) = $s, cos(π/4) = $c")
end

# Test with CPU backend
println("Testing CPU backend:")
set_backend!(CPUBackend())
test_sincos()

# Test with AMDGPU backend (make sure AMDGPU.jl is installed)
println("\nTesting AMDGPU backend:")
set_backend!(AMDGPUBackend())
test_sincos()
