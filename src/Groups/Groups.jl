###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    Groups.jl
### created: Sun Jul 11 18:02:16 2021
###                               


module Groups

abstract type Group end
abstract type Algebra end

include("GroupSU2.jl")

export Group, Algebra
export SU2, SU2alg, dag, normalize, inverse, tr, projalg, norm, norm2
export dot, expm, exp

include("GroupSU3.jl")




end # module
