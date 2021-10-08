###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    MD.jl
### created: Fri Oct  8 21:53:14 2021
###                               

module MD

# Dalla Brida / Luscher coefficients of
# fourth order integrator
const r1 =  0.08398315262876693
const r2 =  0.25397851084105950
const r3 =  0.68223653357190910
const r4 = -0.03230286765269967
const r5 =  0.5-r1-r3
const r6 =  1.0-2.0*(r2+r4)

struct IntrScheme{N, T}
    r::NTuple{N, T}
    eps::T
    ns::Int64    
end

omf4(::Type{T}, eps, ns) where T = IntrScheme{6,T}((r1,r2,r3,r4,r5,r6), eps, ns)
leapfrog(::Type{T}, eps, ns) where T = IntrScheme{2,T}((0.5,1.0,0.5), eps, ns)

export IntrScheme, omf4, leapfrog


end
