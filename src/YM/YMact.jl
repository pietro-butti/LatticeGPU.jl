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

function krnl_impr!(plx, U::AbstractArray{T}, c0, c1, lp::SpaceParm{N,M,D}) where {T,N,M,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    Ush = @cuStaticSharedMem(T, (D,2))
    
    S = zero(eltype(plx))
    for id1 in 1:N-1
        bu1, ru1 = up((b, r), id1, lp)
        Ush[b,1] = U[b,id1,r]

        for id2 = id1+1:N
            bu2, ru2 = up((b, r), id2, lp)
            Ush[b,2] = U[b,id2,r]
            sync_threads()

            # H2 staple
            (b1, r1) = up((b,r), id1, lp)
            if r1 == r
                ga = Ush[b1,1]
            else
                ga = U[b1,id1,r1]
            end
            
            (b2, r2) = up((b1,r1), id1, lp)
            if r2 == r
                gb = Ush[b2,2]
            else
                gb = U[b2,id2,r2]
            end
            
            (b2, r2) = up((b1,r1), id2, lp)
            if r2 == r
                gc = Ush[b2,1]
            else
                gc = U[b2,id1,r2]
            end
            h2 = (ga*gb)/gc
            
            # H3 staple
            (b1, r1) = up((b,r), id2, lp)
            if r1 == r
                ga = Ush[b1,2]
            else
                ga = U[b1,id2,r1]
            end
            
            (b2, r2) = up((b1,r1), id2, lp)
            if r2 == r
                gb = Ush[b2,1]
            else
                gb = U[b2,id1,r2]
            end
            
            (b2, r2) = up((b1,r1), id1, lp)
            if r2 == r
                gc = Ush[b2,2]
            else
                gc = U[b2,id2,r2]
            end
            h3 = (ga*gb)/gc
            # END staples
            
            if ru2 == r
                gb = Ush[bu2,1]
            else
                gb = U[bu2,id1,ru2]
            end
            if ru1 == r
                ga = Ush[bu1,2]
            else
                ga = U[bu1,id2,ru1]
            end

            g2 = Ush[b,2]\Ush[b,1]
            
            S += c0*tr(g2*ga/gb) + c1*( tr(g2*h2/gb) + tr(g2*ga/h3))
        end
    end

    I = point_coord((b,r), lp)
    plx[I] = S

    return nothing
end

function krnl_plaq!(plx, U::AbstractArray{T}, lp::SpaceParm{N,M,D}) where {T,N,M,D}
    
    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    Ush = @cuStaticSharedMem(T, (D,2))
    
    S = zero(eltype(plx))
    for id1 in 1:N-1
        bu1, ru1 = up((b, r), id1, lp)
        Ush[b,1] = U[b,id1,r]

        for id2 = id1+1:N
            bu2, ru2 = up((b, r), id2, lp)
            Ush[b,2] = U[b,id2,r]
            sync_threads()

            if ru1 == r
                gt1 = Ush[bu1,2]
            else
                gt1 = U[bu1,id2,ru1]
            end
            if ru2 == r
                gt2 = Ush[bu2,1]
            else
                gt2 = U[bu2,id1,ru2]
            end
            
            S += tr(Ush[b,1]*gt1 / (Ush[b,2]*gt2))
        end
    end
        
    I = point_coord((b,r), lp)
    plx[I] = S

    return nothing
end

function krnl_force_wilson_pln!(frc1, frc2, U::AbstractArray{T}, ipl, lp::SpaceParm{N,M,D}) where {T,N,M,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    Ush = @cuStaticSharedMem(T, (D,2))
    
    @inbounds begin
        id1, id2 = lp.plidx[ipl]
        bu1, ru1 = up((b, r), id1, lp)
        bu2, ru2 = up((b, r), id2, lp)

        Ush[b,1] = U[b,id1,r]
        Ush[b,2] = U[b,id2,r]
        sync_threads()

        if ru2 == r
            gt2 = Ush[bu2,1]
        else
            gt2 = U[bu2,id1,ru2]
        end
        if ru1 == r
            gt1 = Ush[bu1,2]
        else
            gt1 = U[bu1,id2,ru1]
        end
        
        g1 = gt1/gt2
        g2 = Ush[b,2]\Ush[b,1]
        
        X = projalg(Ush[b,1]*g1/Ush[b,2])
        
        frc1[b  ,id1, r ] -= X
        frc1[b  ,id2, r ] += X
        frc2[bu1,id2,ru1] -= projalg(g1*g2)
        frc2[bu2,id1,ru2] += projalg(g2*g1)
    end
        
    return nothing
end

function krnl_force_impr_pln!(frc1, frc2, U::AbstractArray{T}, c0, c1, ipl, lp::SpaceParm{N,M,D}) where {T,N,M,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    Ush = @cuStaticSharedMem(T, (D,2))
    
    @inbounds begin
        id1, id2 = lp.plidx[ipl]
        bu1, ru1 = up((b, r), id1, lp)
        bu2, ru2 = up((b, r), id2, lp)

        Ush[b,1] = U[b,id1,r]
        Ush[b,2] = U[b,id2,r]
        sync_threads()

        # H1 staple
        (b1, r1) = dw((b,r), id2, lp)
        if r1 == r
            ga = Ush[b1,2]
            gb = Ush[b1,1]
        else
            ga = U[b1,id2,r1]
            gb = U[b1,id1,r1]
        end
        
        (b2, r2) = up((b1,r1), id1, lp)
        if r2 == r
            gc = Ush[b2,2]
        else
            gc = U[b2,id2,r2]
        end
        h1 = (ga\gb)*gc
        
        # H2 staple
        (b1, r1) = up((b,r), id1, lp)
        if r1 == r
            ga = Ush[b1,1]
        else
            ga = U[b1,id1,r1]
        end
        
        (b2, r2) = up((b1,r1), id1, lp)
        if r2 == r
            gb = Ush[b2,2]
        else
            gb = U[b2,id2,r2]
        end
        
        (b2, r2) = up((b1,r1), id2, lp)
        if r2 == r
            gc = Ush[b2,1]
        else
            gc = U[b2,id1,r2]
        end
        h2 = (ga*gb)/gc
        
        # H3 staple
        (b1, r1) = up((b,r), id2, lp)
        if r1 == r
            ga = Ush[b1,2]
        else
            ga = U[b1,id2,r1]
        end
        
        (b2, r2) = up((b1,r1), id2, lp)
        if r2 == r
            gb = Ush[b2,1]
        else
            gb = U[b2,id1,r2]
        end
        
        (b2, r2) = up((b1,r1), id1, lp)
        if r2 == r
            gc = Ush[b2,2]
        else
            gc = U[b2,id2,r2]
        end
        h3 = (ga*gb)/gc
            
        # H4 staple
        (b1, r1) = dw((b,r), id1, lp)
        if r1 == r
            ga = Ush[b1,1]
            gb = Ush[b1,2]
        else
            ga = U[b1,id1,r1]
            gb = U[b1,id2,r1]
        end
        
        (b2, r2) = up((b1,r1), id2, lp)
        if r2 == r
            gc = Ush[b2,1]
        else
            gc = U[b2,id1,r2]
        end
        h4 = (ga\gb)*gc
        # END staples
        
        if ru2 == r
            gb = Ush[bu2,1]
        else
            gb = U[bu2,id1,ru2]
        end
        if ru1 == r
            ga = Ush[bu1,2]
        else
            ga = U[bu1,id2,ru1]
        end
        
        g1 = ga/gb
        g2 = Ush[b,2]\Ush[b,1]

        X = c0*projalg(Ush[b,1]*g1/Ush[b,2]) + c1 * (projalg(Ush[b,1]*h2/(Ush[b,2]*gb)) +
                                                     projalg(Ush[b,1]*ga/(Ush[b,2]*h3)))

        frc1[b,id1,r] -= X + c1*projalg(Ush[b,1]*g1/h4) 
        frc1[b,id2,r] += X + c1*projalg(h1*g1/Ush[b,2]) 

        frc2[bu1,id2,ru1] -= c0*projalg(g1*g2) + c1*( projalg((ga/h3)*g2) +
                                                      projalg((g1/h4)*Ush[b,1]) +
                                                      projalg((g1/Ush[b,2])*h1) )

        frc2[bu2,id1,ru2] += c0*projalg(g2*g1) + c1*( projalg((Ush[b,2]\h1)*g1) +
                                                      projalg(g2*h2/gb) +
                                                      projalg(h4\Ush[b,1]*g1) )
    end
        
    return nothing
end

""" 
    function force_wilson(ymws::YMworkspace, U, lp::SpaceParm)

Computes the force deriving from the Wilson plaquette action, without
the prefactor 1/g0^2, and assign it to the workspace force `ymws.frc1`
"""    
function force_gauge(ymws::YMworkspace, U, c0, lp::SpaceParm)

    if abs(c0-1) < 1.0E-10
        @timeit "Wilson gauge force" begin
            force_wilson_pln!(ymws.frc1, ymws.frc2, U, lp::SpaceParm)
        end
    else
        @timeit "Improved gauge force" begin
            force_wilson_pln!(ymws.frc1, ymws.frc2, U, lp::SpaceParm, c0)
        end
    end
    return nothing
end

force_wilson(ymws::YMworkspace, U, lp::SpaceParm) = force_gauge(ymws, U, 1, lp)


function force_wilson_pln!(frc1, ftmp, U, lp::SpaceParm, c0=1)

    fill!(frc1, zero(eltype(frc1)))
    fill!(ftmp, zero(eltype(ftmp)))
    if c0 == 1
        for i in 1:lp.npls
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_force_wilson_pln!(frc1,ftmp,U,i,lp)
            end
        end
    else
        for i in 1:lp.npls
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_force_impr_pln!(frc1,ftmp,U,c0,(1-c0)/8,i,lp)
            end
        end
    end
    frc1 .= frc1 .+ ftmp
    
    return nothing
end
    
