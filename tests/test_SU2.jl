using LinearAlgebra, Random

import Pkg
#Pkg.activate("/lhome/ific/a/alramos/s.images/julia/workspace/LatticeGPU")
Pkg.activate("/home/alberto/code/julia/LatticeGPU")
using LatticeGPU


T = Float64

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

c = exp(ba, -1.0)
println("Inverse B: ", c)

d = b*c
println("Test:      ", d)

Ma = Array{SU2{T}}(undef, 2)
rand!(Ma)
println(Ma)

fill!(Ma, one(eltype(Ma)))
println(Ma)

println("## Aqui test M2x2")
ba = rand(SU2alg{T})
ga = exp(ba)
println("Matrix: ", alg2mat(ba))
println("Exp:    ", ga)


mo = one(M2x2{T})
println(mo)
mp = mo*ga
println(mp)
println(projalg(mp))
println(projalg(ga))

