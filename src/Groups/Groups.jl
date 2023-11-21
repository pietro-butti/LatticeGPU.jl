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

"""
    abstract type Group

This abstract type encapsulates group types (i.e. \`\`SU(N)\`\`). The following operations/methods are assumed to be defined for a each group.
- `*,/,\\`: Usual group operations.
- [`inverse`](@ref), [`dag`](@ref): The group inverse.
- [`tr`](@ref)
- [`dev_one`](@ref)
- [`unitarize`](@ref)
- [`isgroup`](@ref)
- [`projalg`](@ref)
"""
abstract type Group end

"""
    abstract type Algebra

This abstract type encapsulates algebra types  (i.e. \`\` {\\rm su}(N)\`\`). The following operations/methods are assumed to be defined for each algebra:
- `+,-`: Usual operation of Algebra elements.
- `*,/`: Multiplication/division by Numbers
- `*`: Multiplication of [`Algebra`](@ref) and [`Group`](@ref) or [`GMatrix`](@ref) elements. These routines use the matrix form of the group elements, and return a [`GMatrix`](@ref) type.
- [`exp`](@ref), [`expm`](@ref): Exponential map. Returns a [`Group`](@ref) element.
"""
abstract type Algebra end

"""
    abstract type GM

This abstract type encapsulates \`\` N\\times N \`\` matrix types. The usual matrix operations (`+,-,*,/`) are expected to be defined for this case.
"""
abstract type GMatrix end

export Group, Algebra, GMatrix

##
# SU(2) and 2x2 matrix operations
##
include("SU2Types.jl")
export SU2, SU2alg, M2x2, SU2fund

include("GroupSU2.jl")
include("M2x2.jl")
include("AlgebraSU2.jl")
include("FundamentalSU2.jl")
## END SU(2)

##
# SU(3) and 3x3 matrix operations
##
include("SU3Types.jl")
export SU3, SU3alg, M3x3, SU3fund

include("GroupSU3.jl")
include("M3x3.jl")
include("AlgebraSU3.jl")
include("FundamentalSU3.jl")
export imm, mimm
## END SU(3)


include("AlgebraU2.jl")
include("AlgebraU3.jl")
export U2alg, U3alg

include("GroupU1.jl")
export U1, U1alg


export dot, expm, exp, dag, unitarize, inverse, tr, projalg, norm, norm2, isgroup, alg2mat, dev_one, antsym


end # module
