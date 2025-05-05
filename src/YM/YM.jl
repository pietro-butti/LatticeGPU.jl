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

using CUDA, Random, TimerOutputs, BDIO
using ..Space
using ..Groups
using ..Fields
using ..MD

import Base.show

"""
        struct GaugeParm{T,G,N}

Structure containning the parameters of a pure gauge simulation. These are:
- beta: Type `T`. The bare coupling of the simulation
- c0: Type `T`. LatticeGPU supports the simulation of gauge actions made of 1x1 Wilson Loops and 2x1 Wilson loops. The parameter c0 defines the coefficient on the simulation of the 1x1 loops. Some common choices are:
  - c0=1: Wilson plaquette action
  - c0=5/3: Tree-level improved Lüscher-Weisz action.
  - c0=3.648: Iwasaki gauge action
- cG: Tuple (`T`, `T`). Boundary improvement parameters.
- ng: `Int64`. Rank of the gauge group.
- Ubnd: Boundary field for SF boundary conditions
"""
struct GaugeParm{T,G,N}
    beta::T
    c0::T
    cG::NTuple{2,T}
    ng::Int64

    Ubnd::NTuple{N, G}

    GaugeParm{T1,T2,T3}(a,b,c,d,e) where {T1,T2,T3} = new{T1,T2,T3}(a,b,c,d,e)
    function GaugeParm{T}(::Type{G}, bt, c0, cG, phi, iL) where {T,G}

        degree(::Type{SU2{T}}) where T <: AbstractFloat = 2
        degree(::Type{SU3{T}}) where T <: AbstractFloat = 3
        ng = degree(G)
        nsd = length(iL)
        
        return new{T,G,nsd}(bt, c0, cG, ng, ntuple(id->bndfield(phi[1], phi[2], iL[id]), nsd))
    end
    function GaugeParm{T}(::Type{G}, bt, c0) where {T,G}
        
        degree(::Type{SU2{T}}) where T <: AbstractFloat = 2
        degree(::Type{SU3{T}}) where T <: AbstractFloat = 3
        ng = degree(G)

        return new{T,G,0}(bt, c0, (0.0,0.0), ng, ())
    end
end
export GaugeParm
function Base.show(io::IO, gp::GaugeParm{T, G, N}) where {T,G,N}

    println(io, "Group:  ", G)
    println(io, " - beta:              ", gp.beta)
    println(io, " - c0:                ", gp.c0)
    println(io, " - cG:                ", gp.cG)
    if (N > 0)
        for i in 1:N
            println(io, "   - Boundary link:     ", gp.Ubnd[i])
        end
    end
    
    return nothing
end

"""
        struct YMworkspace{T}

Structure containing memory workspace that is resused by different routines in order to avoid allocating/deallocating time.
The parameter `T` represents the precision of the simulation (i.e. single/double). The structure contains the following components
- GRP: Group being simulated
- ALG: Corresponding Algebra 
- PRC: Precision (i.e. `T`)
- frc1: Algebra field with natural indexing.
- frc2: Algebra field with natural indexing.
- mom:  Algebra field with natural indexing.
- U1:   Group field with natural indexing.
- cm:   Complex field with lexicographic indexing.
- rm:   Real field with lexicographic indexing.
"""
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

"""
        function ztwist(gp::GaugeParm{T,G}, lp::SpaceParm{N,M,B,D}[, ipl])

Returns the twist factor. If a plane index is passed, returns the twist factor as a complex{T}. If this is not provided, returns a tuple, containing the factor of each plane.
"""
function ztwist(gp::GaugeParm{T,G}, lp::SpaceParm{N,M,B,D}) where {T,G,N,M,B,D}

    function plnf(ipl)
        id1, id2 = lp.plidx[ipl]
        return convert(Complex{T},exp(2im * pi * lp.ntw[ipl]/(gp.ng)))
    end

    return ntuple(i->plnf(i), M)
end

function ztwist(gp::GaugeParm{T,G}, lp::SpaceParm{N,M,B,D}, ipl::Int) where {T,G,N,M,B,D}

    id1, id2 = lp.plidx[ipl]
    return convert(Complex{T},exp(2im * pi * lp.ntw[ipl]/(gp.ng)))
end
export ztwist


include("YMfields.jl")
export randomize!, zero!, norm2

include("YMact.jl")
export krnl_plaq!, force_gauge, force_gauge_flw, force_wilson

include("YMhmc.jl")
export gauge_action, hamiltonian, plaquette, HMC!, MD!

include("YMflow.jl")
export FlowIntr, flw, flw_adapt
export Eoft_clover, Eoft_plaq, Qtop
export FlowIntr, wfl_euler, zfl_euler, wfl_rk2, zfl_rk2, wfl_rk3, zfl_rk3

include("YMsf.jl")
export sfcoupling, bndfield, setbndfield

include("YMio.jl")
export import_lex64, import_cern64, import_bsfqcd, save_cnfg, read_cnfg, read_gp

end
