###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    GroupSU3.jl
### created: Sun Jul 11 17:23:02 2021
###                               

#
# Use memory efficient representation: Only store
# first two rows
#
# a.u31 = a.u12*conj(a.u23) - a.u13*conj(a.u22)
# a.u32 = a.u13*conj(a.u21) - a.u11*conj(a.u23)
# a.u33 = a.u11*conj(a.u22) - a.u12*conj(a.u21)
#

import Base.:*, Base.:+, Base.:-,Base.:/,Base.:\
struct SU3 <: Group
    u11::ComplexF64
    u12::ComplexF64
    u13::ComplexF64
    u21::ComplexF64
    u22::ComplexF64
    u23::ComplexF64
end
SU3() = SU3(1,0,0,0,1,0)
inverse(a::SU3) = SU3(conj(a.u11),conj(a.u21),conj(a.u12*conj(a.u23) - a.u13*conj(a.u22)),
                      conj(a.u12),conj(a.u22),conj(a.u13*conj(a.u21) - a.u11*conj(a.u23)))
dag(a::SU3)  = inverse(a)
tr(g::SU3) = a.u11+a.u22+a.u11*conj(a.u22)-a.u12*conj(a.u21)
function Base.:*(a::SU3,b::SU3) 

    bu31 = (b.u12*conj(a.u23) - b.u13*conj(b.u22))
    bu32 = (b.u13*conj(b.u21) - b.u11*conj(b.u23))
    bu33 = (b.u11*conj(b.u22) - b.u12*conj(b.u21))

    return SU3(a.u11*b.u11 + a.u12*b.u21 + a.u13*bu31,
               a.u11*b.u12 + a.u12*b.u22 + a.u13*bu32, 
               a.u11*b.u13 + a.u12*b.u23 + a.u13*bu33, 
               a.u21*b.u11 + a.u22*b.u21 + a.u23*bu31, 
               a.u21*b.u12 + a.u22*b.u22 + a.u23*bu32,
               a.u21*b.u13 + a.u22*b.u23 + a.u23*bu33)
end

