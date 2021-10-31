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

using CUDA, Random, TimerOutputs
using ..Space
using ..Groups
using ..Fields
using ..MD

import Base.show

struct GaugeParm{T,G}
    beta::T
    c0::T
    cG::NTuple{2,T}
    ng::Int64

    Ubnd::G

    function GaugeParm{T}(::Type{G}, bt, c0, cG) where {T,G}

        function degree(::Type{SU2{T}}) where T
            return 2
        end
        function degree(::Type{SU3{T}}) where T <: AbstractFloat
            return 3
        end
        ng = degree(G)
        
        return new{T,G}(bt, c0, cG, ng, one(G))
    end
    function GaugeParm{T}(::Type{G}, bt, c0) where {T,G}
        
        degree(::Type{SU2{T}}) where T <: AbstractFloat = 2
        degree(::Type{SU3{T}}) where T <: AbstractFloat = 3
        ng = degree(G)

        return new{T,G}(bt, c0, (0.0,0.0), ng, one(G))
    end
end
export GaugeParm
function Base.show(io::IO, gp::GaugeParm{T, G}) where {T,G}

    println(io, "Group:  ", G)
    println(io, " - beta:              ", gp.beta)
    println(io, " - c0:                ", gp.c0)
    println(io, " - cG:                ", gp.cG)
    println(io, " - Boundary link:     ", gp.Ubnd)
    
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
        
        @timeit "Allocating YMWorkspace" begin
            if (G == SU2)
                GRP = SU2
                ALG = SU2alg
                f1 = vector_field(SU2alg{T}, lp)
                f2 = vector_field(SU2alg{T}, lp)
                mm = vector_field(SU2alg{T}, lp)
                u1 = vector_field(SU2{T},    lp)
            end
            
            if (G == SU3)
                GRP = SU3
                ALG = SU3alg
                f1 = vector_field(SU3alg{T}, lp)
                f2 = vector_field(SU3alg{T}, lp)
                mm = vector_field(SU3alg{T}, lp)
                u1 = vector_field(SU3{T},    lp)
            end
            cs = scalar_field_point(Complex{T}, lp)
            rs = scalar_field_point(T, lp)
        end
            
        return new{T}(GRP,ALG,T,f1, f2, mm, u1, cs, rs)
    end
end
export YMworkspace
function Base.show(io::IO, ymws::YMworkspace)
    
    println(io, "Workspace for Group:   ", ymws.GRP)
    println(io, "              Algebra: ", ymws.ALG)
    println(io, "Precision:             ", ymws.PRC)

    return nothing
end


function ztwist(gp::GaugeParm{T,G}, lp::SpaceParm{N,M,B,D}) where {T,G,N,M,B,D}

    function plnf(ipl)
        id1, id2 = lp.plidx[ipl]
        return convert(Complex{T},exp(2im * pi * lp.ntw[ipl]/(lp.iL[id1]*lp.iL[id2]*gp.ng)))
    end

    return ntuple(i->plnf(i), M)
end

function ztwist(gp::GaugeParm{T,G}, lp::SpaceParm{N,M,B,D}, ipl::Int) where {T,G,N,M,B,D}

    id1, id2 = lp.plidx[ipl]
    return convert(Complex{T},exp(2im * pi * lp.ntw[ipl]/(lp.iL[id1]*lp.iL[id2]*gp.ng)))
end
export ztwist


include("YMfields.jl")
export randomize!, zero!, norm2

include("YMact.jl")
export krnl_plaq!, force0_wilson!

include("YMhmc.jl")
export gauge_action, hamiltonian, plaquette, HMC!, OMF4!

include("YMflow.jl")
export wfl_euler, wfl_rk3, zfl_euler, zfl_rk3, Eoft_clover, Eoft_plaq, Qtop

include("YMsf.jl")
export sfcoupling

end
