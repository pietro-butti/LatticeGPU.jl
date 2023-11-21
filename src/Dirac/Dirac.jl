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


struct DiracParam{T,R}
    m0::T
    csw::T
    th::NTuple{4,Complex{T}}
    ct::T


    function DiracParam{T}(::Type{R},m0,csw,th,ct) where {T,R} 
        return new{T,R}(m0,csw,th,ct)
    end

    function DiracParam{T}(m0,csw,th,ct) where {T} 
        return new{T,SU3fund}(m0,csw,th,ct)
    end
end
function Base.show(io::IO, dpar::DiracParam{T,R}) where {T,R}

    println(io, "Wilson fermions in the: ", R, " representation")
    println(io, " - Bare mass: ", dpar.m0," // Kappa = ",0.5/(dpar.m0+4))
    println(io, " - Csw :                ", dpar.csw)
    println(io, " - c_t:                 ", dpar.ct)
    println(io, " - Theta:               ", dpar.th)
    return nothing
end

struct DiracWorkspace{T}
    sr
    sp
    sAp
    st
    csw
    
    function DiracWorkspace(::Type{G}, ::Type{T}, lp::SpaceParm{4,6,B,D}) where {G,T <: AbstractFloat, B,D}

        @timeit "Allocating DiracWorkspace" begin
            if G == SU3fund
                sr  = scalar_field(Spinor{4,G}, lp)
                sp  = scalar_field(Spinor{4,G}, lp)
                sAp = scalar_field(Spinor{4,G}, lp)
                st  = scalar_field(Spinor{4,G}, lp)
                csw = tensor_field(U3alg{T},lp)
            end
            
            if G == SU2fund
                sr  = scalar_field(Spinor{4,G}, lp)
                sp  = scalar_field(Spinor{4,G}, lp)
                sAp = scalar_field(Spinor{4,G}, lp)
                st  = scalar_field(Spinor{4,G}, lp)
                csw = tensor_field(U2alg{T},lp)
            end
        end

        return new{T}(sr,sp,sAp,st,csw,cs)
        end

    end
end

export DiracWorkspace, DiracParam


"""
    function Csw!(dws, U, gp, lp::SpaceParm)

Computes the clover and stores it in dws.csw.

"""
function Csw!(dws, U, gp, lp::SpaceParm{4,6,B,D}) where {B,D}

    @timeit "Csw computation" begin

        for i in 1:Int(lp.npls)
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_csw!(dws.csw, U, gp.Ubnd, i, lp)
            end
        end
    end
        
    return nothing
end

function krnl_csw!(csw::AbstractArray{T}, U, Ubnd, ipl, lp::SpaceParm{4,M,B,D}) where {T,M,B,D}
        
    @inbounds begin
        b = Int64(CUDA.threadIdx().x)
        r = Int64(CUDA.blockIdx().x)
        I = point_coord((b,r), lp)
        it = I[4]

        id1, id2 = lp.plidx[ipl]
        SFBC = ((B == BC_SF_AFWB) || (B == BC_SF_ORBI) ) && (id1 == 4)
        
        bu1, ru1 = up((b, r), id1, lp)
        bu2, ru2 = up((b, r), id2, lp)
        bd1, rd1 = dw((b, r), id1, lp)
        bd2, rd2 = dw((b, r), id2, lp)
        bdd, rdd = dw((bd1, rd1), id2, lp)
        bud, rud = dw((bu1, ru1), id2, lp)
        bdu, rdu = up((bd1, rd1), id2, lp)
        
        if SFBC && (it == lp.iL[end])
            gt1 = Ubnd[id2]
            gt2 = Ubnd[id2]
        else
            gt1 = U[bu1,id2,ru1]
            gt2 = U[bud,id2,rud]
        end

        M1 = U[b,id1,r]*gt1/(U[b,id2,r]*U[bu2,id1,ru2])
        M2 = (U[bd2,id2,rd2]\(U[bd2,id1,rd2]*gt2))/U[b,id1,r]
        M3 = (U[bdd,id2,rdd]*U[bd1,id1,rd1])\(U[bdd,id1,rdd]*U[bd2,id2,rd2])
        M4 = (U[b,id2,r]/(U[bd1,id2,rd1]*U[bdu,id1,rdu]))*U[bd1,id1,rd1]


        if !(SFBC && (it == 1))
            csw[b,ipl,r]  = 0.125*(antsym(M1)+antsym(M2)+antsym(M3)+antsym(M4))
        end
  
    end
        
    return nothing
end

function Dw!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::SpaceParm{4,6,B,D}) where {B,D}
       
    if abs(dpar.csw) > 1.0E-10
        @timeit "Dw" begin
            CUDA.@sync begin
               CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_Dwimpr!(so, U, si, dws.csw, dpar.m0, dpar.th, dpar.csw, lp)
            end
        end
    else
        @timeit "Dw" begin
            CUDA.@sync begin
               CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_Dw!(so, U, si, dpar.m0, dpar.th, lp)
            end
        end
    end
    
    return nothing
end

function krnl_Dwimpr!(so, U, si, Fcsw, m0, th, csw, lp::SpaceParm{4,6,B,D}) where {B,D}

    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    bu1, ru1 = up((b,r), 1, lp)
    bd1, rd1 = dw((b,r), 1, lp)
    bu2, ru2 = up((b,r), 2, lp)
    bd2, rd2 = dw((b,r), 2, lp)
    bu3, ru3 = up((b,r), 3, lp)
    bd3, rd3 = dw((b,r), 3, lp)
    bu4, ru4 = up((b,r), 4, lp)
    bd4, rd4 = dw((b,r), 4, lp)

    @inbounds begin 
        
        so[b,r] = (4+m0)*si[b,r]  + 0.5*csw*im*( Fcsw[b,1,r]*dmul(Gamma{10},si[b,r]) + Fcsw[b,2,r]*dmul(Gamma{11},si[b,r]) + Fcsw[b,3,r]*dmul(Gamma{12},si[b,r])
                                                +Fcsw[b,4,r]*dmul(Gamma{15},si[b,r]) + Fcsw[b,5,r]*dmul(Gamma{14},si[b,r]) + Fcsw[b,6,r]*dmul(Gamma{13},si[b,r]))          

        so[b,r] -= 0.5*(th[1]*gpmul(Pgamma{1,-1},U[b,1,r],si[bu1,ru1]) +conj(th[1])*gdagpmul(Pgamma{1,+1},U[bd1,1,rd1],si[bd1,rd1]) +
                        th[2]*gpmul(Pgamma{2,-1},U[b,2,r],si[bu2,ru2]) +conj(th[2])*gdagpmul(Pgamma{2,+1},U[bd2,2,rd2],si[bd2,rd2]) +
                        th[3]*gpmul(Pgamma{3,-1},U[b,3,r],si[bu3,ru3]) +conj(th[3])*gdagpmul(Pgamma{3,+1},U[bd3,3,rd3],si[bd3,rd3]) +
                        th[4]*gpmul(Pgamma{4,-1},U[b,4,r],si[bu4,ru4]) +conj(th[4])*gdagpmul(Pgamma{4,+1},U[bd4,4,rd4],si[bd4,rd4]) )

    end

    return nothing
end

function krnl_Dw!(so, U, si, m0, th, lp::SpaceParm{4,6,B,D}) where {B,D}

    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    bu1, ru1 = up((b,r), 1, lp)
    bd1, rd1 = dw((b,r), 1, lp)
    bu2, ru2 = up((b,r), 2, lp)
    bd2, rd2 = dw((b,r), 2, lp)
    bu3, ru3 = up((b,r), 3, lp)
    bd3, rd3 = dw((b,r), 3, lp)
    bu4, ru4 = up((b,r), 4, lp)
    bd4, rd4 = dw((b,r), 4, lp)

    @inbounds begin 
        
        so[b,r] = (4+m0)*si[b,r]

        so[b,r] -= 0.5*(th[1]*gpmul(Pgamma{1,-1},U[b,1,r],si[bu1,ru1]) +conj(th[1])*gdagpmul(Pgamma{1,+1},U[bd1,1,rd1],si[bd1,rd1]) +
                        th[2]*gpmul(Pgamma{2,-1},U[b,2,r],si[bu2,ru2]) +conj(th[2])*gdagpmul(Pgamma{2,+1},U[bd2,2,rd2],si[bd2,rd2]) +
                        th[3]*gpmul(Pgamma{3,-1},U[b,3,r],si[bu3,ru3]) +conj(th[3])*gdagpmul(Pgamma{3,+1},U[bd3,3,rd3],si[bd3,rd3]) +
                        th[4]*gpmul(Pgamma{4,-1},U[b,4,r],si[bu4,ru4]) +conj(th[4])*gdagpmul(Pgamma{4,+1},U[bd4,4,rd4],si[bd4,rd4]) )

    end

    return nothing
end

function Dw!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::Union{SpaceParm{4,6,BC_SF_ORBI,D},SpaceParm{4,6,BC_SF_AFWB,D}}) where {D}
       
    if abs(dpar.csw) > 1.0E-10
        @timeit "Dw" begin
            CUDA.@sync begin
               CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_Dwimpr!(so, U, si, dws.csw, dpar.m0, dpar.th, dpar.csw, dpar.ct, lp)
            end
        end
    else
        @timeit "Dw" begin
            CUDA.@sync begin
               CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_Dw!(so, U, si, dpar.m0, dpar.th, dpar.ct, lp)
            end
        end
    end
    
    return nothing
end

function krnl_Dwimpr!(so, U, si, Fcsw, m0, th, csw, ct, lp::Union{SpaceParm{4,6,BC_SF_ORBI,D},SpaceParm{4,6,BC_SF_AFWB,D}}) where {D}

    # The field si is assumed to be zero at t = 0

    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    if (point_time((b,r),lp) != 1)

        bu1, ru1 = up((b,r), 1, lp)
        bd1, rd1 = dw((b,r), 1, lp)
        bu2, ru2 = up((b,r), 2, lp)
        bd2, rd2 = dw((b,r), 2, lp)
        bu3, ru3 = up((b,r), 3, lp)
        bd3, rd3 = dw((b,r), 3, lp)
        bu4, ru4 = up((b,r), 4, lp)
        bd4, rd4 = dw((b,r), 4, lp)

        @inbounds begin 
            
            so[b,r] = (4+m0)*si[b,r]  + 0.5*csw*im*( Fcsw[b,1,r]*dmul(Gamma{10},si[b,r]) + Fcsw[b,2,r]*dmul(Gamma{11},si[b,r]) + Fcsw[b,3,r]*dmul(Gamma{12},si[b,r])
                                                    +Fcsw[b,4,r]*dmul(Gamma{15},si[b,r]) + Fcsw[b,5,r]*dmul(Gamma{14},si[b,r]) + Fcsw[b,6,r]*dmul(Gamma{13},si[b,r]))          

            
            so[b,r] -= 0.5*(th[1]*gpmul(Pgamma{1,-1},U[b,1,r],si[bu1,ru1]) +conj(th[1])*gdagpmul(Pgamma{1,+1},U[bd1,1,rd1],si[bd1,rd1]) +
                            th[2]*gpmul(Pgamma{2,-1},U[b,2,r],si[bu2,ru2]) +conj(th[2])*gdagpmul(Pgamma{2,+1},U[bd2,2,rd2],si[bd2,rd2]) +
                            th[3]*gpmul(Pgamma{3,-1},U[b,3,r],si[bu3,ru3]) +conj(th[3])*gdagpmul(Pgamma{3,+1},U[bd3,3,rd3],si[bd3,rd3]) +
                            th[4]*gpmul(Pgamma{4,-1},U[b,4,r],si[bu4,ru4]) +conj(th[4])*gdagpmul(Pgamma{4,+1},U[bd4,4,rd4],si[bd4,rd4]) )

                if (point_time((b,r),lp) == 2) || (point_time((b,r),lp) == lp.iL[4])
                    so[b,r] += (ct-1.0)*si[b,r]
                end
        end
    end

    return nothing
end

function krnl_Dw!(so, U, si, m0, th, ct, lp::Union{SpaceParm{4,6,BC_SF_ORBI,D},SpaceParm{4,6,BC_SF_AFWB,D}}) where {D}

    # The field si is assumed to be zero at t = 0

    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    if (point_time((b,r),lp) != 1)
            
        bu1, ru1 = up((b,r), 1, lp)
        bd1, rd1 = dw((b,r), 1, lp)
        bu2, ru2 = up((b,r), 2, lp)
        bd2, rd2 = dw((b,r), 2, lp)
        bu3, ru3 = up((b,r), 3, lp)
        bd3, rd3 = dw((b,r), 3, lp)
        bu4, ru4 = up((b,r), 4, lp)
        bd4, rd4 = dw((b,r), 4, lp)

        @inbounds begin 
            
            so[b,r] = (4+m0)*si[b,r]
            so[b,r] -= 0.5*(th[1]*gpmul(Pgamma{1,-1},U[b,1,r],si[bu1,ru1]) +conj(th[1])*gdagpmul(Pgamma{1,+1},U[bd1,1,rd1],si[bd1,rd1]) +
                            th[2]*gpmul(Pgamma{2,-1},U[b,2,r],si[bu2,ru2]) +conj(th[2])*gdagpmul(Pgamma{2,+1},U[bd2,2,rd2],si[bd2,rd2]) +
                            th[3]*gpmul(Pgamma{3,-1},U[b,3,r],si[bu3,ru3]) +conj(th[3])*gdagpmul(Pgamma{3,+1},U[bd3,3,rd3],si[bd3,rd3]) +
                            th[4]*gpmul(Pgamma{4,-1},U[b,4,r],si[bu4,ru4]) +conj(th[4])*gdagpmul(Pgamma{4,+1},U[bd4,4,rd4],si[bd4,rd4]) )

            if (point_time((b,r),lp) == 2) || (point_time((b,r),lp) == lp.iL[4])
                so[b,r] += (ct-1.0)*si[b,r]
            end
        end
    end

    return nothing
end

function g5Dw!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::SpaceParm{4,6,B,D}) where {B,D}
       
    if abs(dpar.csw) > 1.0E-10
        @timeit "g5Dw" begin
            CUDA.@sync begin
               CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dwimpr!(so, U, si, dws.csw, dpar.m0, dpar.th, dpar.csw, lp)
            end
        end
    else
        @timeit "g5Dw" begin
            CUDA.@sync begin
               CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dw!(so, U, si, dpar.m0, dpar.th, lp)
            end
        end
    end
    
    return nothing
end

function krnl_g5Dwimpr!(so, U, si, Fcsw, m0, th, csw, lp::SpaceParm{4,6,B,D}) where {B,D}

    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    bu1, ru1 = up((b,r), 1, lp)
    bd1, rd1 = dw((b,r), 1, lp)
    bu2, ru2 = up((b,r), 2, lp)
    bd2, rd2 = dw((b,r), 2, lp)
    bu3, ru3 = up((b,r), 3, lp)
    bd3, rd3 = dw((b,r), 3, lp)
    bu4, ru4 = up((b,r), 4, lp)
    bd4, rd4 = dw((b,r), 4, lp)

    @inbounds begin 
        
        so[b,r] = (4+m0)*si[b,r]  + 0.5*csw*im*( Fcsw[b,1,r]*dmul(Gamma{10},si[b,r]) + Fcsw[b,2,r]*dmul(Gamma{11},si[b,r]) + Fcsw[b,3,r]*dmul(Gamma{12},si[b,r])
                                                +Fcsw[b,4,r]*dmul(Gamma{15},si[b,r]) + Fcsw[b,5,r]*dmul(Gamma{14},si[b,r]) + Fcsw[b,6,r]*dmul(Gamma{13},si[b,r]))          

        so[b,r] -= 0.5*(th[1]*gpmul(Pgamma{1,-1},U[b,1,r],si[bu1,ru1]) +conj(th[1])*gdagpmul(Pgamma{1,+1},U[bd1,1,rd1],si[bd1,rd1]) +
                        th[2]*gpmul(Pgamma{2,-1},U[b,2,r],si[bu2,ru2]) +conj(th[2])*gdagpmul(Pgamma{2,+1},U[bd2,2,rd2],si[bd2,rd2]) +
                        th[3]*gpmul(Pgamma{3,-1},U[b,3,r],si[bu3,ru3]) +conj(th[3])*gdagpmul(Pgamma{3,+1},U[bd3,3,rd3],si[bd3,rd3]) +
                        th[4]*gpmul(Pgamma{4,-1},U[b,4,r],si[bu4,ru4]) +conj(th[4])*gdagpmul(Pgamma{4,+1},U[bd4,4,rd4],si[bd4,rd4]) )

        so[b,r] = dmul(Gamma{5}, so[b,r])
    end

    return nothing
end

function krnl_g5Dw!(so, U, si, m0, th, lp::SpaceParm{4,6,B,D}) where {B,D}

    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    bu1, ru1 = up((b,r), 1, lp)
    bd1, rd1 = dw((b,r), 1, lp)
    bu2, ru2 = up((b,r), 2, lp)
    bd2, rd2 = dw((b,r), 2, lp)
    bu3, ru3 = up((b,r), 3, lp)
    bd3, rd3 = dw((b,r), 3, lp)
    bu4, ru4 = up((b,r), 4, lp)
    bd4, rd4 = dw((b,r), 4, lp)

    @inbounds begin 
        
        so[b,r] = (4+m0)*si[b,r]

        so[b,r] -= 0.5*(th[1]*gpmul(Pgamma{1,-1},U[b,1,r],si[bu1,ru1]) +conj(th[1])*gdagpmul(Pgamma{1,+1},U[bd1,1,rd1],si[bd1,rd1]) +
                        th[2]*gpmul(Pgamma{2,-1},U[b,2,r],si[bu2,ru2]) +conj(th[2])*gdagpmul(Pgamma{2,+1},U[bd2,2,rd2],si[bd2,rd2]) +
                        th[3]*gpmul(Pgamma{3,-1},U[b,3,r],si[bu3,ru3]) +conj(th[3])*gdagpmul(Pgamma{3,+1},U[bd3,3,rd3],si[bd3,rd3]) +
                        th[4]*gpmul(Pgamma{4,-1},U[b,4,r],si[bu4,ru4]) +conj(th[4])*gdagpmul(Pgamma{4,+1},U[bd4,4,rd4],si[bd4,rd4]) )

        so[b,r] = dmul(Gamma{5}, so[b,r])
    end

    return nothing
end

function g5Dw!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::Union{SpaceParm{4,6,BC_SF_ORBI,D},SpaceParm{4,6,BC_SF_AFWB,D}}) where {D}
       
    if abs(dpar.csw) > 1.0E-10
        @timeit "g5Dw" begin
            CUDA.@sync begin
               CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dwimpr!(so, U, si, dws.csw, dpar.m0, dpar.th, dpar.csw, dpar.ct, lp)
            end
        end
    else
        @timeit "g5Dw" begin
            CUDA.@sync begin
               CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dw!(so, U, si, dpar.m0, dpar.th, dpar.ct, lp)
            end
        end
    end
    
    return nothing
end

function krnl_g5Dwimpr!(so, U, si, Fcsw, m0, th, csw, ct, lp::Union{SpaceParm{4,6,BC_SF_ORBI,D},SpaceParm{4,6,BC_SF_AFWB,D}}) where {D}

    # The field si is assumed to be zero at t = 0

    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    if (point_time((b,r),lp) != 1)

        bu1, ru1 = up((b,r), 1, lp)
        bd1, rd1 = dw((b,r), 1, lp)
        bu2, ru2 = up((b,r), 2, lp)
        bd2, rd2 = dw((b,r), 2, lp)
        bu3, ru3 = up((b,r), 3, lp)
        bd3, rd3 = dw((b,r), 3, lp)
        bu4, ru4 = up((b,r), 4, lp)
        bd4, rd4 = dw((b,r), 4, lp)

        @inbounds begin 
            
            so[b,r] = (4+m0)*si[b,r]  + 0.5*csw*im*( Fcsw[b,1,r]*dmul(Gamma{10},si[b,r]) + Fcsw[b,2,r]*dmul(Gamma{11},si[b,r]) + Fcsw[b,3,r]*dmul(Gamma{12},si[b,r])
                                                    +Fcsw[b,4,r]*dmul(Gamma{15},si[b,r]) + Fcsw[b,5,r]*dmul(Gamma{14},si[b,r]) + Fcsw[b,6,r]*dmul(Gamma{13},si[b,r]))          

            
            so[b,r] -= 0.5*(th[1]*gpmul(Pgamma{1,-1},U[b,1,r],si[bu1,ru1]) +conj(th[1])*gdagpmul(Pgamma{1,+1},U[bd1,1,rd1],si[bd1,rd1]) +
                            th[2]*gpmul(Pgamma{2,-1},U[b,2,r],si[bu2,ru2]) +conj(th[2])*gdagpmul(Pgamma{2,+1},U[bd2,2,rd2],si[bd2,rd2]) +
                            th[3]*gpmul(Pgamma{3,-1},U[b,3,r],si[bu3,ru3]) +conj(th[3])*gdagpmul(Pgamma{3,+1},U[bd3,3,rd3],si[bd3,rd3]) +
                            th[4]*gpmul(Pgamma{4,-1},U[b,4,r],si[bu4,ru4]) +conj(th[4])*gdagpmul(Pgamma{4,+1},U[bd4,4,rd4],si[bd4,rd4]) )

                if (point_time((b,r),lp) == 2) || (point_time((b,r),lp) == lp.iL[4])
                    so[b,r] += (ct-1.0)*si[b,r]
                end
        end
    end

    so[b,r] = dmul(Gamma{5}, so[b,r])

    return nothing
end

function krnl_g5Dw!(so, U, si, m0, th, ct, lp::Union{SpaceParm{4,6,BC_SF_ORBI,D},SpaceParm{4,6,BC_SF_AFWB,D}}) where {D}

    # The field si is assumed to be zero at t = 0

    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    if (point_time((b,r),lp) != 1)
            
        bu1, ru1 = up((b,r), 1, lp)
        bd1, rd1 = dw((b,r), 1, lp)
        bu2, ru2 = up((b,r), 2, lp)
        bd2, rd2 = dw((b,r), 2, lp)
        bu3, ru3 = up((b,r), 3, lp)
        bd3, rd3 = dw((b,r), 3, lp)
        bu4, ru4 = up((b,r), 4, lp)
        bd4, rd4 = dw((b,r), 4, lp)

        @inbounds begin 
            
            so[b,r] = (4+m0)*si[b,r]
            so[b,r] -= 0.5*(th[1]*gpmul(Pgamma{1,-1},U[b,1,r],si[bu1,ru1]) +conj(th[1])*gdagpmul(Pgamma{1,+1},U[bd1,1,rd1],si[bd1,rd1]) +
                            th[2]*gpmul(Pgamma{2,-1},U[b,2,r],si[bu2,ru2]) +conj(th[2])*gdagpmul(Pgamma{2,+1},U[bd2,2,rd2],si[bd2,rd2]) +
                            th[3]*gpmul(Pgamma{3,-1},U[b,3,r],si[bu3,ru3]) +conj(th[3])*gdagpmul(Pgamma{3,+1},U[bd3,3,rd3],si[bd3,rd3]) +
                            th[4]*gpmul(Pgamma{4,-1},U[b,4,r],si[bu4,ru4]) +conj(th[4])*gdagpmul(Pgamma{4,+1},U[bd4,4,rd4],si[bd4,rd4]) )

            if (point_time((b,r),lp) == 2) || (point_time((b,r),lp) == lp.iL[4])
                so[b,r] += (ct-1.0)*si[b,r]
            end
        end
    end

    so[b,r] = dmul(Gamma{5}, so[b,r])

    return nothing
end

function DwdagDw!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::Union{SpaceParm{4,6,BC_SF_ORBI,D},SpaceParm{4,6,BC_SF_AFWB,D}}) where {D}

    if abs(dpar.csw) > 1.0E-10
        @timeit "DwdagDw" begin
            
            @timeit "g5Dw" begin
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dwimpr!(dws.st, U, si, dws.csw, dpar.m0, dpar.th, dpar.csw, dpar.ct, lp)
                end
            end

            @timeit "g5Dw" begin
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dwimpr!(so, U, dws.st, dws.csw, dpar.m0, dpar.th, dpar.csw, dpar.ct, lp)
                end
            end
        end
    else
        @timeit "DwdagDw" begin
            
            @timeit "g5Dw" begin
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dw!(dws.st, U, si, dpar.m0, dpar.th, dpar.ct, lp)
                end
            end

            @timeit "g5Dw" begin
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dw!(so, U, dws.st, dpar.m0, dpar.th, dpar.ct, lp)
                end
            end
        end
    end

    return nothing
end

function DwdagDw!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::SpaceParm{4,6,B,D}) where {B,D}

    if abs(dpar.csw) > 1.0E-10
        @timeit "DwdagDw" begin
            
            @timeit "g5Dw" begin
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dwimpr!(dws.st, U, si, dws.csw, dpar.m0, dpar.th, dpar.csw, lp)
                end
            end

            @timeit "g5Dw" begin
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dwimpr!(so, U, dws.st, dws.csw, dpar.m0, dpar.th, dpar.csw, lp)
                end
            end
        end
    else
        @timeit "DwdagDw" begin
            
            @timeit "g5Dw" begin
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dw!(dws.st, U, si, dpar.m0, dpar.th, lp)
                end
            end

            @timeit "g5Dw" begin
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dw!(so, U, dws.st, dpar.m0, dpar.th, lp)
                end
            end
        end
    end
    
    return nothing
end


function SF_bndfix!(sp, lp::Union{SpaceParm{4,6,BC_SF_ORBI,D},SpaceParm{4,6,BC_SF_AFWB,D}}) where {D}
    CUDA.@sync begin
        CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_sfbndfix!(sp, lp)
    end
    
    return nothing
end

function krnl_sfbndfix!(sp,lp::SpaceParm)
    b=Int64(CUDA.threadIdx().x)
    r=Int64(CUDA.blockIdx().x)

    if (point_time((b,r),lp) == 1)
        sp[b,r] = 0.0*sp[b,r]
    end
    return nothing
end


"""
    function pfrandomize!(f::AbstractArray{Spinor{4, SU3fund / SU2fund {T}}}, lp::SpaceParm, t::Int64 = 0)

Randomizes the SU2fund / SU3fund fermion field. If the argument t is present, it only randomizes that time-slice. 
"""
function pfrandomize!(f::AbstractArray{Spinor{4, SU3fund{T}}}, lp::SpaceParm, t::Int64 = 0) where {T} 

    @timeit "Randomize pseudofermion field" begin
        p = ntuple(i->CUDA.randn(T, lp.bsz, 3, lp.rsz,2),4) # complex generation not suported for Julia 1.5.4
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_assign_pf_su3!(f,p,lp,t)
        end
    end

    return nothing
end

function krnl_assign_pf_su3!(f::AbstractArray, p , lp::SpaceParm, t::Int64)

    @inbounds begin
        b = Int64(CUDA.threadIdx().x)
        r = Int64(CUDA.blockIdx().x)

            if t == 0
            f[b,r] = Spinor(map(x->SU3fund(x[b,1,r,1] + im* x[b,1,r,2],
                                        x[b,2,r,1] + im* x[b,2,r,2],
                                        x[b,3,r,1] + im* x[b,3,r,2]),p))
            elseif point_time((b,r),lp) == t
            f[b,r] = Spinor(map(x->SU3fund(x[b,1,r,1] + im* x[b,1,r,2],
                                        x[b,2,r,1] + im* x[b,2,r,2],
                                        x[b,3,r,1] + im* x[b,3,r,2]),p))
            end
            
    end

    return nothing
end

function pfrandomize!(f::AbstractArray{Spinor{4, SU2fund{T}}},lp::SpaceParm, t::Int64=0) where {T} 

    @timeit "Randomize pseudofermion field" begin
        p = ntuple(i->CUDA.randn(T, lp.bsz, 2, lp.rsz,2),4) # complex generation not suported for Julia 1.5.4
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_assign_pf_su2!(f,p,lp,t)
        end
    end

    return nothing
end

function krnl_assign_pf_su2!(f::AbstractArray, p , lp::SpaceParm, t::Int64)

    @inbounds begin
        b = Int64(CUDA.threadIdx().x)
        r = Int64(CUDA.blockIdx().x)

            if t == 0
            f[b,r] = Spinor(map(x->SU2fund(x[b,1,r,1] + im* x[b,1,r,2],
                                        x[b,2,r,1] + im* x[b,2,r,2]),p))
            elseif point_time((b,r),lp) == t
            f[b,r] = Spinor(map(x->SU2fund(x[b,1,r,1] + im* x[b,1,r,2],
                                        x[b,2,r,1] + im* x[b,2,r,2]),p))
            end
            
    end

    return nothing
end

export Dw!, g5Dw!, DwdagDw!, SF_bndfix!, Csw!, pfrandomize!


include("DiracIO.jl")
export read_prop, save_prop, read_dpar


end
