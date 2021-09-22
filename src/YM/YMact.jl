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

function krnl_force_wilson_nw!(fpl, U::AbstractArray{T}, lp::SpaceParm{N,M,D}) where {T,N,M,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    Ush = @cuStaticSharedMem(T, (D,N))

    for id in 1:N
        Ush[b,id] = U[b,id,r]
    end
    sync_threads()
    
    @inbounds for ipl in 1:lp.npls
        id1, id2 = lp.plidx[ipl]

        bu1, ru1 = up((b, r), id1, lp)
        bu2, ru2 = up((b, r), id2, lp)
            
        if ru2 == r
            gt2 = Ush[bu2,id1]
        else
            gt2 = U[bu2,id1,ru2]
        end
        if ru1 == r
            gt1 = Ush[bu1,id2]
        else
            gt1 = U[bu1,id2,ru1]
        end
        
        g1 = gt1/gt2
        g2 = Ush[b,id2]\Ush[b,id1]
        
        fpl[b  ,ipl, r  ,1] = projalg(Ush[b,id1]*g1/Ush[b,id2])
        fpl[bu1,ipl, ru1,2] = projalg(g1*g2)
        fpl[bu2,ipl, ru2,3] = projalg(g2*g1)
        
    end
        
    return nothing
end

function krnl_add_force_plns!(frc::AbstractArray{T}, fpl, lp::SpaceParm{N,M,D}) where {T,N,M,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    Fsh = @cuStaticSharedMem(T, (D,M))
    @inbounds for j in 1:M
        Fsh[b,j] = fpl[b,j,r,1]
    end
    sync_threads()
    
    @inbounds for ipl in 1:lp.npls
        id1, id2 = lp.plidx[ipl]

        frc[b,id1,r] += -Fsh[b,ipl] + fpl[b,ipl,r,3]
        frc[b,id2,r] +=  Fsh[b,ipl] - fpl[b,ipl,r,2]
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
        CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_force_wilson_nw!(fpln,U,lp)
    end
    fill!(frc1, zero(eltype(frc1)))
    CUDA.@sync begin
        CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_force_plns!(frc1, fpln,lp)
    end

    return nothing
end
