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

function Dw!(so, U, si, m0, lp::SpaceParm)

    @timeit "Dw" begin
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_Dw!(so, U, si, m0, [1,1,1,1], lp)
        end
    end
    
    return nothing
end

function DwdagDw!(so, U, si, m0, lp::SpaceParm)

    @timeit "DwdagDw" begin
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dw!(st, U, si, m0, [1,1,1,1], lp)
        end
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dw!(so, U, st, m0, [1,1,1,1], lp)
        end
    end
    
    return nothing
end


function krnl_Dw!(so, U, si, m0, th, lp::SpaceParm{4,6,B,D}) where {B,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

<    # For SF:
    #  - cttilde affects mass term at x0 = a, T-a
    #  - Spinor can be periodic as long as 0 at x_0=0
    so[b,r] = (4+m0)*si[b,r]
    for id in 1:4
        bu, ru = up((b,r), id, lp)
        bd, rd = dw((b,r), id, lp)

        so[b,r] -= ( th[id]*gpmul(Pgamma{id,-1},U[b,id,r],si[bu,ru]) +
            conj(th[id])*gdagpmul(Pgamma{id,+1},U[bd,id,rd],si[bd,rd]) )/2
    end

    return nothing
end

function krnl_g5Dw!(so, U, si, m0, th, lp::SpaceParm{4,6,B,D}) where {B,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
    
    so[b,r] = (4+m0)*si[b,r]
    for id in 1:4
        bu, ru = up((b,r), id, lp)
        bd, rd = dw((b,r), id, lp)

        so[b,r] -= ( th[id]*gpmul(Pgamma{id,-1},U[b,id,r],si[bu,ru]) +
            conj(th[id])*gdagpmul(Pgamma{id,+1},U[bd,id,rd],si[bd,rd]) )/2
    end
    so[b,r].s[3] = -so[b,r].s[3] 
    so[b,r].s[4] = -so[b,r].s[4] 

    return nothing
end

export Dw

end
