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

export Group, Algebra

include("GroupSU2.jl")
export SU2, SU2alg

include("GroupSU3.jl")
export SU3, SU3alg

export dot, expm, exp, dag, normalize, inverse, tr, projalg, norm, norm2, isgroup


end # module
