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

using CUDA, Random
import Base.:*, Base.:+, Base.:-,Base.:/,Base.:\,Base.exp,Base.one,Base.zero
import Random.rand

abstract type Group end
abstract type Algebra end

export Group, Algebra

##
# SU(2) and 2x2 matrix operations
##
include("SU2Types.jl")
export SU2, SU2alg, M2x2

include("GroupSU2.jl")
include("M2x2.jl")
include("AlgebraSU2.jl")
## END SU(2)

##
# SU(3) and 3x3 matrix operations
##
include("SU3Types.jl")
export SU3, SU3alg, M3x3

include("GroupSU3.jl")
include("M3x3.jl")
include("AlgebraSU3.jl")
## END SU(3)

include("GroupU1.jl")
export U1, U1alg


export dot, expm, exp, dag, normalize, inverse, tr, projalg, norm, norm2, isgroup, alg2mat


end # module
