using LinearAlgebra, Random

import Pkg
#Pkg.activate("/lhome/ific/a/alramos/s.images/julia/workspace/LatticeGPU")
Pkg.activate("/home/alberto/code/julia/LatticeGPU")
using LatticeGPU


T = Float32

b = rand(SU2{T})
println(b)

ba = rand(SU2alg{T})
println("Ba:        ", ba)
b = exp(ba)
println("B:         ", b)
println(typeof(norm2(ba)))

c = inverse(b)
println("Inverse B: ", c)

d = b*c
println("Test:      ", d)

Ma = Array{SU2{T}}(undef, 100)
rand!(Ma)
println(Ma)

fill!(Ma, one(eltype(Ma)))
println(Ma)
