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

function krnl_impr!(plx, U::AbstractArray{T}, c0, c1, Ubnd::T, cG, ztw, lp::SpaceParm{N,M,B,D}) where {T,N,M,B,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
    it = point_time((b, r), lp)

    Ush = @cuStaticSharedMem(T, (D,2))

    ipl = 0
    S = zero(eltype(plx))
    for id1 in N:-1:1
        bu1, ru1 = up((b, r), id1, lp)
        SFBC  = ((B == BC_SF_AFWB) || (B == BC_SF_ORBI) ) && (id1==N) 
        Ush[b,1] = U[b,id1,r]

        for id2 = 1:id1-1
            bu2, ru2 = up((b, r), id2, lp)
            Ush[b,2] = U[b,id2,r]
            sync_threads()
            ipl = ipl + 1
            
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
                if SFBC && (it == lp.iL[end])
                    gc = Ubnd
                else
                    gc = U[b2,id2,r2]
                end
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
                if SFBC && (it == lp.iL[end])
                    ga = Ubnd
                else
                    ga = U[bu1,id2,ru1]
                end
            end

            g2 = Ush[b,2]\Ush[b,1]
            
            if (it == lp.iL[end]) && SFBC
                S += cG*(c0*tr(g2*ga/gb) + (3*c1/2)*tr(g2*ga/h3))
            elseif (it == 1) && SFBC
                S += cG*(c0*tr(g2*ga/gb) + (3*c1/2)*tr(g2*ga/h3)) + c1*tr(g2*h2/gb)
            else
                S += ztw[ipl]*c0*tr(g2*ga/gb) +
                    (ztw[ipl]^2*c1)*( tr(g2*h2/gb) + tr(g2*ga/h3))
            end

        end
    end

    I = point_coord((b,r), lp)
    plx[I] = S

    return nothing
end

function krnl_plaq!(plx, U::AbstractArray{T}, Ubnd::T, cG, ztw, lp::SpaceParm{N,M,B,D}) where {T,N,M,B,D}
    
    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
    it = point_time((b, r), lp)

    Ush = @cuStaticSharedMem(T, (D,2))
    
    S = zero(eltype(plx))
    ipl = 0
    for id1 in N:-1:1
        bu1, ru1 = up((b, r), id1, lp)
        Ush[b,1] = U[b,id1,r]

        SFBND = ( ( (B == BC_SF_AFWB) || (B == BC_SF_ORBI) ) &&
                  ( (it == 1) || (it == lp.iL[end])) ) && (id1 == N) 

        for id2 = 1:id1-1
            bu2, ru2 = up((b, r), id2, lp)
            Ush[b,2] = U[b,id2,r]
            sync_threads()
            ipl = ipl + 1
            
            if ru1 == r
                gt1 = Ush[bu1,2]
            else
                if SFBND && (it == lp.iL[end])
                    gt1 = Ubnd
                else
                    gt1 = U[bu1,id2,ru1]
                end
            end
            if ru2 == r
                gt2 = Ush[bu2,1]
            else
                gt2 = U[bu2,id1,ru2]
            end

            if SFBND
                S += cG*tr(Ush[b,1]*gt1 / (Ush[b,2]*gt2))
            else
                S += ztw[ipl]*tr(Ush[b,1]*gt1 / (Ush[b,2]*gt2))
            end
        end
    end
        
    I = point_coord((b,r), lp)
    plx[I] = S

    return nothing
end

function krnl_force_wilson_pln!(frc1, frc2, U::AbstractArray{T}, Ubnd::T, cG, ztw, ipl, lp::SpaceParm{N,M,B,D}) where {T,N,M,B,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
    it = point_time((b,r), lp)

    Ush = @cuStaticSharedMem(T, (D,2))
    
    @inbounds begin
        id1, id2 = lp.plidx[ipl]
        bu1, ru1 = up((b, r), id1, lp)
        bu2, ru2 = up((b, r), id2, lp)

        SFBC = ((B == BC_SF_AFWB) || (B == BC_SF_ORBI) ) && (id1 == N)

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
            if SFBC && (it == lp.iL[end])
                gt1 = Ubnd
            else
                gt1 = U[bu1,id2,ru1]
            end
        end
        
        g1 = gt1/gt2
        g2 = Ush[b,2]\Ush[b,1]

        if SFBC && (it == 1)
            X = cG*projalg(ztw,Ush[b,1]*g1/Ush[b,2])
            
            frc1[b  ,id1, r ] -= X
            frc2[bu1,id2,ru1] -= cG*projalg(ztw,g1*g2)
            frc2[bu2,id1,ru2] += cG*projalg(ztw,g2*g1)
        elseif SFBC && (it == lp.iL[end])
            X = cG*projalg(ztw,Ush[b,1]*g1/Ush[b,2])
            
            frc1[b  ,id1, r ] -= X
            frc1[b  ,id2, r ] += X
            frc2[bu2,id1,ru2] += cG*projalg(ztw,g2*g1)
        else
            X = projalg(ztw,Ush[b,1]*g1/Ush[b,2])
            
            frc1[b  ,id1, r ] -= X
            frc1[b  ,id2, r ] += X
            frc2[bu1,id2,ru1] -= projalg(ztw,g1*g2)
            frc2[bu2,id1,ru2] += projalg(ztw,g2*g1)
        end
    end
    
    return nothing
end

function krnl_force_impr_pln!(frc1, frc2, U::AbstractArray{T}, c0, c1, Ubnd::T, cG, ztw, ipl, lp::SpaceParm{N,M,B,D}) where {T,N,M,B,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x
    it = point_time((b, r), lp)

    Ush = @cuStaticSharedMem(T, (D,2))
    
    @inbounds begin
        id1, id2 = lp.plidx[ipl]
        bu1, ru1 = up((b, r), id1, lp)
        bu2, ru2 = up((b, r), id2, lp)

        SFBC = ((B == BC_SF_AFWB) || (B == BC_SF_ORBI) ) && (id1 == N)

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
            if SFBC && (it == lp.iL[end])
                gc = Ubnd
            else
                gc = U[b2,id2,r2]
            end
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
            if SFBC && (it == lp.iL[end])
                gc = Ubnd
            else
                gc = U[b2,id2,r2]
            end
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
            if SFBC && (it == lp.iL[end])
                ga = Ubnd
            else
                ga = U[bu1,id2,ru1]
            end
        end
        
        g1 = ga/gb
        g2 = Ush[b,2]\Ush[b,1]

        if SFBC && (it == 1)
            X = (cG*c0)*projalg(Ush[b,1]*g1/Ush[b,2]) + c1*projalg(Ush[b,1]*h2/(Ush[b,2]*gb)) +
                (3*c1*cG/2)*projalg(Ush[b,1]*ga/(Ush[b,2]*h3)) 
            
            frc1[b,id1,r] -= X
            
            frc2[bu1,id2,ru1] -= (cG*c0)*projalg(g1*g2) + (3*c1*cG/2)*projalg((ga/h3)*g2) +
                (3*c1*cG/2)*projalg((g1/Ush[b,2])*h1)
            
            frc2[bu2,id1,ru2] += (cG*c0)*projalg(g2*g1) + (3*c1*cG/2) * projalg((Ush[b,2]\h1)*g1) +
                c1*projalg(g2*h2/gb) 
        elseif SFBC && (it == lp.iL[end])
            X = (cG*c0)*projalg(Ush[b,1]*g1/Ush[b,2]) +
                (3*c1*cG/2) * (projalg(Ush[b,1]*ga/(Ush[b,2]*h3))) 
            
            frc1[b,id1,r] -= X + c1*projalg(Ush[b,1]*g1/h4) 
            frc1[b,id2,r] += X + (3*c1*cG/2)*projalg(h1*g1/Ush[b,2]) 
            
            frc2[bu2,id1,ru2] += (cG*c0)*projalg(g2*g1) + (3*c1*cG/2) * projalg((Ush[b,2]\h1)*g1) +
                c1 * projalg(h4\Ush[b,1]*g1) 
        else
            zsq = ztw[ipl]^2
            X = projalg(c0*ztw[ipl],Ush[b,1]*g1/Ush[b,2]) + projalg(zsq*c1,Ush[b,1]*h2/(Ush[b,2]*gb)) +
                projalg(zsq*c1,Ush[b,1]*ga/(Ush[b,2]*h3))
            
            frc1[b,id1,r] -= X + projalg(zsq*c1,Ush[b,1]*g1/h4) 
            frc1[b,id2,r] += X + projalg(zsq*c1,h1*g1/Ush[b,2]) 
            
            frc2[bu1,id2,ru1] -= projalg(c0*ztw[ipl],g1*g2) + projalg(zsq*c1,(ga/h3)*g2) +
                projalg(zsq*c1,(g1/h4)*Ush[b,1]) + projalg(zsq*c1,(g1/Ush[b,2])*h1) 
            
            frc2[bu2,id1,ru2] += projalg(c0*ztw[ipl],g2*g1) + projalg(zsq*c1,(Ush[b,2]\h1)*g1) +
                projalg(zsq*c1,g2*h2/gb) + projalg(zsq*c1,h4\Ush[b,1]*g1) 
        end
            
    end
        
    return nothing
end

""" 
    function force_wilson(ymws::YMworkspace, U, lp::SpaceParm)

Computes the force deriving from the Wilson plaquette action, without
the prefactor 1/g0^2, and assign it to the workspace force `ymws.frc1`
"""    
function force_gauge(ymws::YMworkspace, U, c0, cG, gp::GaugeParm, lp::SpaceParm)

    ztw = ztwist(gp, lp)
    if abs(c0-1) < 1.0E-10
        @timeit "Wilson gauge force" begin
            force_pln!(ymws.frc1, ymws.frc2, U, gp.Ubnd, cG, ztw, lp::SpaceParm)
        end
    else
        @timeit "Improved gauge force" begin
            force_pln!(ymws.frc1, ymws.frc2, U, gp.Ubnd, cG, ztw, lp::SpaceParm, c0)
        end
    end
    return nothing
end

force_gauge(ymws::YMworkspace, U, c0, gp, lp) = force_gauge(ymws, U, c0, gp.cG[1], gp, lp)
force_wilson(ymws::YMworkspace, U, gp::GaugeParm, lp::SpaceParm) = force_gauge(ymws, U, 1, gp, lp)
force_wilson(ymws::YMworkspace, U, cG, gp::GaugeParm, lp::SpaceParm) = force_gauge(ymws, U, 1, gp.cG[1], gp, lp)


function force_pln!(frc1, ftmp, U, Ubnd, cG, ztw, lp::SpaceParm, c0=1)

    fill!(frc1, zero(eltype(frc1)))
    fill!(ftmp, zero(eltype(ftmp)))
    if c0 == 1
        for i in 1:lp.npls
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_force_wilson_pln!(frc1,ftmp,U, Ubnd, cG, ztw[i], i,lp)
            end
        end
    else
        for i in 1:lp.npls
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_force_impr_pln!(frc1,ftmp,U,c0,(1-c0)/8,Ubnd, cG, ztw[i], i,lp)
            end
        end
    end
    frc1 .= frc1 .+ ftmp
    
    return nothing
end
    
