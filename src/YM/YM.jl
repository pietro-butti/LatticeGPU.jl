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

import Base.show

struct GaugeParm{T}
    beta::T
    c0::T
    cG::NTuple{2,T}
    ng::Int64
end
export GaugeParm
function Base.show(io::IO, gp::GaugeParm)

    println(io, "beta:   ", gp.beta)
    println(io, "c0:     ", gp.c0)
    println(io, "Ngauge: ", gp.ng)
    
    return nothing
end

struct YMworkspace{T}
    GRP
    ALG
    PRC
    frc1
    frc2
    mom
    U1
    cm # complex of volume
    rm # float   of volume
    function YMworkspace(::Type{G}, ::Type{T}, lp::SpaceParm) where {G <: Group, T <: AbstractFloat}
        
        if (G == SU2)
            GRP = SU2
            ALG = SU2alg
            f1 = field(SU2alg{T}, lp)
            f2 = field(SU2alg{T}, lp)
            mm = field(SU2alg{T}, lp)
            u1 = field(SU2{T},    lp)
        end

        if (G == SU3)
            GRP = SU3
            ALG = SU3alg
            f1 = field(SU3alg{T}, lp)
            f2 = field(SU3alg{T}, lp)
            mm = field(SU3alg{T}, lp)
            u1 = field(SU3{T},    lp)
        end
        cs = CuArray{Complex{T}, 2}(undef, lp.bsz,lp.rsz)
        rs = CuArray{T, 2}(undef, lp.bsz,lp.rsz)

        return new{T}(GRP,ALG,T,f1, f2, mm, u1, cs, rs)
    end
end
export YMworkspace
function Base.show(io::IO, ymws::YMworkspace)
    
    println(io, "Workspace for Group:   ", ymws.GRP)
    println(io, "              Algebra: ", ymws.ALG)
    println(io, "Precision:             ", ymws.PRC)
    if ymws.fpln == nothing
        println(io, "  - Running in memory efficient mode")
    else
        println(io, "  - Running in computing efficient mode")
    end
    return nothing
end


include("YMfields.jl")
export field, field_pln, randomize!, zero!, norm2

include("YMact.jl")
export krnl_plaq!, force0_wilson!

include("YMhmc.jl")
export gauge_action, hamiltonian, plaquette, HMC!, OMF4!

include("YMflow.jl")
export wfl_euler, wfl_rk3

end
