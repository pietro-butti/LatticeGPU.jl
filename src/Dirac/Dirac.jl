###
### "THE BEER-WARE LICENSE":
### Alberto Ramos and Carlos Pena wrote this file. As long as you retain this  
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy us a beer in 
### return. <alberto.ramos@cern.ch> <carlos.pena@uam.es>
###
### file:    Dirac.jl
### created: Thu Nov 18 17:20:24 2021
###                               


module Dirac

using CUDA, TimerOutputs
using ..Space
using ..Groups
using ..Fields
using ..YM
using ..Spinors

"""
    struct DiracParam{T,R}

Stores the parameters of the Dirac operator. It can be generated via the constructor `function DiracParam{T}(::Type{R},m0,csw,th,tm,ct)`. The first argument can be ommited and is taken to be `SU3fund`.
The parameters are:

- `m0::T` : Mass of the fermion
- `csw::T` : Improvement coefficient for the Csw term
- `th{Ntuple{4,Complex{T}}}` : Phase for the fermions included in the boundary conditions, reabsorbed in the Dirac operator.
- `tm` : Twisted mass parameter
- `ct` : Boundary improvement term, only used for Schrödinger Funtional boundary conditions. 
"""
struct DiracParam{T,R}
    m0::T
    csw::T
    th::NTuple{4,Complex{T}}
    tm::T
    ct::T

    function DiracParam{T}(::Type{R},m0,csw,th,tm,ct) where {T,R}
        return new{T,R}(m0,csw,th,tm,ct)
    end

    function DiracParam{T}(m0,csw,th,tm,ct) where {T}
        return new{T,SU3fund}(m0,csw,th,tm,ct)
    end
end
function Base.show(io::IO, dpar::DiracParam{T,R}) where {T,R}

    println(io, "Wilson fermions in the: ", R, " representation")
    println(io, " - Bare mass: ", dpar.m0," // Kappa = ",0.5/(dpar.m0+4))
    println(io, " - Csw :                ", dpar.csw)
    println(io, " - Theta:               ", dpar.th)
    println(io, " - Twisted mass:        ", dpar.tm)
    println(io, " - c_t:                 ", dpar.ct)
    return nothing
end


"""
    struct DiracWorkspace{T}    

Workspace needed to work with fermion fields. It contains four scalar fermion fields and, for the SU2fund and SU3fund, a U(N) field to store the clover term.

It can be created with the constructor `DiracWorkspace(::Type{G}, ::Type{T}, lp::SpaceParm{4,6,B,D})`. For example:

dws = DiracWorkspace(SU2fund,Float64,lp);
dws = DiracWorkspace(SU3fund,Float64,lp);

"""
struct DiracWorkspace{T}
    sr
    sp
    sAp
    st
    csw
    
    function DiracWorkspace(::Type{G}, ::Type{T}, lp::SpaceParm{4,6,B,D}) where {G,T <: AbstractFloat, B,D}

        @timeit "Allocating DiracWorkspace" begin
            if G == SU3fund
                sr  = scalar_field(Spinor{4,SU3fund{T}}, lp)
                sp  = scalar_field(Spinor{4,SU3fund{T}}, lp)
                sAp = scalar_field(Spinor{4,SU3fund{T}}, lp)
                st  = scalar_field(Spinor{4,SU3fund{T}}, lp)
                csw = tensor_field(U3alg{T},lp)
            elseif G == SU2fund
                sr  = scalar_field(Spinor{4,SU2fund{T}}, lp)
                sp  = scalar_field(Spinor{4,SU2fund{T}}, lp)
                sAp = scalar_field(Spinor{4,SU2fund{T}}, lp)
                st  = scalar_field(Spinor{4,SU2fund{T}}, lp)
                csw = tensor_field(U2alg{T},lp)
            else
                sr  = scalar_field(Spinor{4,G}, lp)
                sp  = scalar_field(Spinor{4,G}, lp)
                sAp = scalar_field(Spinor{4,G}, lp)
                st  = scalar_field(Spinor{4,G}, lp)
                csw = nothing
            end
        end

        return new{T}(sr,sp,sAp,st,csw)
        end

end


"""
    function mtwmdpar(dpar::DiracParam)

Returns `dpar` with oposite value of the twisted mass.
"""
function mtwmdpar(dpar::DiracParam{P,R}) where {P,R}
    return DiracParam{P}(R,dpar.m0,dpar.csw,dpar.th,-dpar.tm,dpar.ct)
end


export DiracWorkspace, DiracParam, mtwmdpar

include("Diracfields.jl")
export SF_bndfix!, Csw!, pfrandomize!

include("Diracoper.jl")
export Dw!, g5Dw!, DwdagDw!

include("DiracIO.jl")
export read_prop, save_prop, read_dpar

include("Diracflow.jl")
export Nablanabla!, Dslash_sq!, flw, backflow


end
