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
    
    function DiracWorkspace(::Type{G}, ::Type{T}, lp::SpaceParm{4,6,B,D}) where {G,T <: AbstractFloat, B,D}

        sr  = scalar_field(Spinor{4,G}, lp)
        sp  = scalar_field(Spinor{4,G}, lp)
        sAp = scalar_field(Spinor{4,G}, lp)
        st  = scalar_field(Spinor{4,G}, lp)
        return new{T}(sr,sp,sAp,st)
    end
end
export DiracWorkspace, DiracParam

function Dw!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::SpaceParm{4,6,B,D}) where {B,D}
     
        @timeit "Dw" begin
            CUDA.@sync begin
               CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_Dw!(so, U, si, dpar.m0, dpar.th, lp)
            end
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
       
        @timeit "Dw" begin
            CUDA.@sync begin
               CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_Dw!(so, U, si, dpar.m0, dpar.th, dpar.ct, lp)
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
       
        @timeit "g5Dw" begin
            CUDA.@sync begin
               CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dw!(so, U, si, dpar.m0, dpar.th, lp)
            end
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
       
        @timeit "g5Dw" begin
            CUDA.@sync begin
               CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dw!(so, U, si, dpar.m0, dpar.th, dpar.ct, lp)
            end
        end
    
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

function DwdagDw!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::SpaceParm{4,6,B,D}) where {B,D}

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

    return nothing
end

function DwdagDw!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::Union{SpaceParm{4,6,BC_SF_ORBI,D},SpaceParm{4,6,BC_SF_AFWB,D}}) where {D}

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
    return nothing
end


export Dw!, g5Dw!, DwdagDw!

end
