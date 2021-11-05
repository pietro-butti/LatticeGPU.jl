###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    LatticeGPU.jl
### created: Sat Jul 17 17:19:58 2021
###                               


module LatticeGPU

include("Groups/Groups.jl")

using .Groups
export Group, Algebra
export SU2, SU2alg, SU3, SU3alg, M3x3, M2x2, U1, U1alg
export dot, expm, exp, dag, unitarize, inverse, tr, projalg, norm, norm2, isgroup, alg2mat, dev_one

include("Space/Space.jl")

using .Space
export SpaceParm
export up, dw, updw, point_coord, point_index, point_time
export BC_PERIODIC, BC_OPEN, BC_SF_AFWB, BC_SF_ORBI

include("Fields/Fields.jl")
using .Fields
export vector_field, scalar_field, nscalar_field, scalar_field_point

include("MD/MD.jl")
using .MD
export IntrScheme
export omf4, leapfrog, omf2

include("YM/YM.jl")

using .YM
export ztwist
export YMworkspace, GaugeParm, force0_wilson!, field, field_pln, randomize!, zero!, norm2
export gauge_action, hamiltonian, plaquette, HMC!, OMF4!
export Eoft_clover, Eoft_plaq, Qtop
export FlowIntr, wfl_euler, zfl_euler, wfl_rk2, zfl_rk2, wfl_rk3, zfl_rk3
export flw, flw_adapt
export sfcoupling, bndfield, setbndfield

end # module
