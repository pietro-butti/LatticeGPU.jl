###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    YM.jl
### created: Mon Jul 12 16:23:51 2021
###                               


module YM

using CUDA, Random, StructArrays
using ..Space
using ..Groups

struct GaugeParm
    beta::Float64
    cG::Tuple{Float64,Float64}
    ng::Int32
end
export GaugeParm

include("YMfields.jl")
export field, randomn!, zero!, norm2

struct YMworkspace
    frc1
    frc2
    mom
    U1
    cm # complex of volume
    function YMworkspace(::Type{T}, lp::SpaceParm) where {T <: Union{Group,Algebra}}
        
        if (T == SU2)
            f1 = field(SU2alg, lp)
            f2 = field(SU2alg, lp)
            mm = field(SU2alg, lp)
            u1 = field(SU2,    lp)
            cs = zeros(ComplexF64,lp.iL...)
            rs = zeros(Float64,   lp.iL...)
            return new(f1, f2, mm, u1, replace_storage(CuArray, cs))
        end
        return nothing
    end
end
export YMworkspace

include("YMact.jl")
export krnl_plaq!, force0_wilson!

include("YMhmc.jl")
export gauge_action, hamiltonian, HMC!, OMF4!

end
