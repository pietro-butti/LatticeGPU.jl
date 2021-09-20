###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    YMact.jl
### created: Mon Jul 12 18:31:19 2021
###                               

function krnl_plaq!(plx, U, ipl, lp::SpaceParm)

    id1, id2 = lp.plidx[ipl]
    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
    bu1, ru1 = up((b, r), id1, lp)
    bu2, ru2 = up((b, r), id2, lp)

    @inbounds plx[b, r] = tr(U[b,id1,r]*U[bu1,id2,ru1] / (U[b,id2,r]*U[bu2,id1,ru2]))

    return nothing
end

function krnl_plaq!(plx, U, lp::SpaceParm)
    
    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
    plx[b,r] = zero(plx[b,r])
    @inbounds for ipl in 1:lp.npls
        id1, id2 = lp.plidx[ipl]
        
        bu1, ru1 = up((b, r), id1, lp)
        bu2, ru2 = up((b, r), id2, lp)
                    
        plx[b,r] += tr(U[b,id1,r]*U[bu1,id2,ru1] / (U[b,id2,r]*U[bu2,id1,ru2]))
    end
    
    return nothing
end

function krnl_force_wilson_pln!(frc1, frc2, U, ipl, lp::SpaceParm)

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    @inbounds begin
        id1, id2 = lp.plidx[ipl]
        bu1, ru1 = up((b, r), id1, lp)
        bu2, ru2 = up((b, r), id2, lp)
        
        g1 = U[bu1,id2,ru1]/U[bu2,id1,ru2]
        g2 = U[b,id2,r]\U[b,id1,r]
        
        F1 = projalg(U[b,id1,r]*g1/U[b,id2,r])
        F2 = projalg(g1*g2)
        F3 = projalg(g2*g1)
        
        frc1[b  ,id1, r ] -= F1
        frc1[b  ,id2, r ] += F1
        frc2[bu1,id2,ru1] -= F2
        frc2[bu2,id1,ru2] += F3
    end
        
    return nothing
end

function krnl_force_wilson_nw!(fpl, U, lp::SpaceParm)

    b   = CUDA.threadIdx().x
    ipl = mod1(CUDA.blockIdx().x, lp.npls)
    r   = div(CUDA.blockIdx().x-1, lp.npls)+1

    @inbounds begin #for ipl in 1:lp.npls
        id1, id2 = lp.plidx[ipl]
        bu1, ru1 = up((b, r), id1, lp)
        bu2, ru2 = up((b, r), id2, lp)
        
        g1 = U[bu1,id2,ru1]/U[bu2,id1,ru2]
        g2 = U[b,id2,r]\U[b,id1,r]
        
        F1 = projalg(U[b,id1,r]*g1/U[b,id2,r])
        F2 = projalg(g1*g2)
        F3 = projalg(g2*g1)
        
        fpl[b  ,ipl, r  ,1,1] = -F1
        fpl[b  ,ipl, r  ,2,1] =  F1
        fpl[bu1,ipl, ru1,2,2] = -F2
        fpl[bu2,ipl, ru2,1,2] =  F3
    end
        
    return nothing
end



function krnl_force_wilson!(frc, U, lp::SpaceParm)

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    @inbounds for id1 in 1:lp.ndim
        bu1, ru1, bd1, rd1 = updw((b, r), id1, lp)
        @inbounds for id2 in id1+1:lp.ndim
            bu2, ru2, bd2, rd2 = updw((b, r), id2, lp)
            bud, rud = dw((bu1,ru1), id2, lp)
            bdu, rdu = dw((bu2,ru2), id1, lp)
            
            F1 = projalg(U[b,id1,r]*U[bu1,id2,ru1]/(U[b,id2,r]*U[bu2,id1,ru2]))
            F2 = projalg((U[b,id2,r]/(U[bd1,id2,rd1]*U[bdu,id1,rdu]))*U[bd1,id1,rd1])
            F3 = projalg((U[bd2,id2,rd2]\U[bd2,id1,rd2])*(U[bud,id2,rud]/U[b,id1,r]))
            
            frc[b,id1,r] -= F1 - F3
            frc[b,id2,r] += F1 - F2
        end
    end
        
    return nothing
end

function krnl_add_force_plns!(frc, fpl, lp::SpaceParm)

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
    @inbounds for ipl in 1:lp.npls
        id1, id2 = lp.plidx[ipl]
        frc[b,id1,r] += fpl[b,ipl,r,1,1] + fpl[b,ipl,r,1,2]
        frc[b,id2,r] += fpl[b,ipl,r,2,1] + fpl[b,ipl,r,2,2]
    end
        
    return nothing
end

function force0_wilson_pln!(frc1, ftmp, U, lp::SpaceParm)

    fill!(frc1, zero(eltype(frc1)))
    fill!(ftmp, zero(eltype(ftmp)))
    for i in 1:lp.npls
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_force_wilson_pln!(frc1,ftmp,U,i,lp)
        end
    end
    frc1 .= frc1 .+ ftmp
    
    return nothing
end
    
function force0_wilson!(frc1, U, lp::SpaceParm)

    fill!(frc1, zero(eltype(frc1)))
    CUDA.@sync begin
        CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_force_wilson!(frc1,U,lp)
    end
    
    return nothing
end

function force0_wilson_nw!(frc1, fpln, U, lp::SpaceParm)

    CUDA.@sync begin
        CUDA.@cuda threads=lp.bsz blocks=lp.rsz*lp.npls krnl_force_wilson_nw!(fpln,U,lp)
    end
    fill!(frc1, zero(eltype(frc1)))
    CUDA.@sync begin
        CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_force_plns!(frc1, fpln,lp)
    end

    return nothing
end
