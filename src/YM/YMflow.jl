###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    YMflow.jl
### created: Sat Sep 25 08:37:14 2021
###                               

"""
    function add_zth_term(ymws::YMworkspace, U, lp)

Assuming that the gauge improved (LW) force is in ymws.frc1, this routine
adds the "Zeuthen term" and returns the full zeuthen force in ymws.frc1
"""
function add_zth_term(ymws::YMworkspace, U, lp)

    CUDA.@sync begin
        CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_zth!(ymws.frc1,ymws.frc2,U,lp)
    end
    ymws.frc1 .= ymws.frc2 
    
    return nothing
end

function krnl_add_zth!(frc, frc2::AbstractArray{TA}, U::AbstractArray{TG}, lp::SpaceParm{N,M,B,D}) where {TA,TG,N,M,B,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    Ush = @cuStaticSharedMem(TG, D)
    Fsh = @cuStaticSharedMem(TA, D)
    
    @inbounds for id in 1:N
        Ush[b] = U[b,id,r]
        Fsh[b] = frc[b,id,r]
        sync_threads()
        
        bu, ru = up((b,r), id, lp)
        bd, rd = dw((b,r), id, lp)
        
        if ru == r
            X = Fsh[bu]
        else
            X = frc[bu,id,ru]
        end
        if rd == r
            Y  = Fsh[bd]
            Ud = Ush[bd]
        else
            Y  = frc[bd,id,rd]
            Ud = U[bd,id,rd]
        end
        
        frc2[b,id,r] = (5/6)*Fsh[b] + (1/6)*(projalg(Ud\Y*Ud) +
                                             projalg(Ush[b]*X/Ush[b]))
    end
    
    return nothing
end


function flw_euler(U, ns, eps, c0, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace; add_zth=false)
    
    @timeit "Integrating flow equations (Euler)" begin
        for i in 1:ns
            force_gauge(ymws, U, c0, 1, gp, lp)
            if add_zth
                add_zth_term(ymws::YMworkspace, U, lp)
            end
            U .= expm.(U, ymws.frc1, 2*eps)
        end
    end
    
    return nothing
end

function flw_rk3(U, ns, eps, c0, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace; add_zth=false)
    
    @timeit "Integrating flow equations (RK3)" begin
        for i in 1:ns
            e0 = eps/2
            force_gauge(ymws, U, c0, 1, gp, lp)
            if add_zth
                add_zth_term(ymws::YMworkspace, U, lp)
            end
            ymws.mom .= ymws.frc1
            U .= expm.(U, ymws.mom, e0)
            
            e0 = -34*eps/36
            e1 = 16*eps/9
            force_gauge(ymws, U, c0, 1, gp, lp)
            if add_zth
                add_zth_term(ymws::YMworkspace, U, lp)
            end
            ymws.mom .= e0.*ymws.mom .+ e1.*ymws.frc1
            U .= expm.(U, ymws.mom)
            
            e1 = 6*eps/4
            force_gauge(ymws, U, c0, 1, gp, lp)
            if add_zth
                add_zth_term(ymws::YMworkspace, U, lp)
            end
            ymws.mom .= e1.*ymws.frc1 .- ymws.mom 
            U .= expm.(U, ymws.mom)
        end
    end
                              
    return nothing
end



                              
wfl_euler(U, ns, eps, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace) = flw_euler(U, ns, eps, 1, gp, lp, ymws)
zfl_euler(U, ns, eps, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace) = flw_euler(U, ns, eps, 5.0/3.0, gp, lp, ymws, add_zth=true)
wfl_rk3(U, ns, eps, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace) = flw_rk3(U, ns, eps, 1, gp, lp, ymws)
zfl_rk3(U, ns, eps, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace) = flw_rk3(U, ns, eps, 5.0/3.0, gp, lp, ymws, add_zth=true)


##
# Observables
##

"""
    function Eoft_plaq([Eslc,] U, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace)

Measure the action density `E(t)` using the plaquette discretization. If the argument `Eslc`
the contribution for each Euclidean time slice and plane are returned.
"""
function Eoft_plaq(Eslc, U, gp::GaugeParm{T}, lp::SpaceParm{N,M,B,D}, ymws::YMworkspace) where {T,N,M,B,D}

    @timeit "E(t) plaquette measurement" begin

        tp = ntuple(i->i, N-1)
        V3 = prod(lp.iL[1:end-1])

        fill!(Eslc,zero(T))
        Etmp = zeros(T,lp.iL[end])
        for ipl in 1:M
            fill!(Etmp, zero(T))
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_plaq_pln!(ymws.cm, U, ipl, lp)
            end
            
            Etmp .=  (gp.ng .- reshape(Array(CUDA.mapreduce(real, +, ymws.cm;dims=tp)),lp.iL[end])/V3 )
            if ipl < N
                for it in 2:lp.iL[end]
                    Eslc[it,ipl] = Etmp[it] + Etmp[it-1]
                end
                Eslc[1,ipl] = Etmp[1] + Etmp[end]
            else
                for it in 1:lp.iL[end]
                    Eslc[it,ipl] = 2*Etmp[it]
                end
            end
        end
        
    end


    return sum(Eslc)/lp.iL[end]
end

Eoft_plaq(U, gp::GaugeParm{T}, lp::SpaceParm{N,M,B,D}, ymws::YMworkspace) where {T,N,M,B,D} = Eoft_plaq(zeros(T,lp.iL[end],M), U, gp, lp, ymws)


function krnl_plaq_pln!(plx, U::AbstractArray{T}, ipl, lp::SpaceParm{N,M,B,D}) where {T,N,M,B,D}
    
    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    id1, id2 = lp.plidx[ipl]

    bu1, ru1 = up((b, r), id1, lp)
    bu2, ru2 = up((b, r), id2, lp)

    I = point_coord((b,r), lp)
    plx[I] = tr(U[b,1,r]*U[bu1,id2,ru1] / (U[b,2,r]*U[bu2,id1,ru2]))

    return nothing
end

"""
    Qtop([Qslc,] U, lp, ymws)

Measure the topological charge `Q` of the configuration `U`. If the argument `Qslc` is present
the contribution for each Euclidean time slice are returned.
"""
function Qtop(Qslc, U, lp::SpaceParm{4,M,B,D}, ymws::YMworkspace) where {M,B,D}

    @timeit "Qtop measurement" begin

        tp = (1,2,3)
        
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor!(ymws.frc1, ymws.frc2, U, 1,5, lp)
        end
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_qd!(ymws.rm, +, ymws.frc1, ymws.frc2, U, lp)
        end
    
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor!(ymws.frc1, ymws.frc2, U, 2,4, lp)
        end
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_qd!(ymws.rm, -, ymws.frc1, ymws.frc2, U, lp)
        end
    
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor!(ymws.frc1, ymws.frc2, U, 3,6, lp)
        end
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_qd!(ymws.rm, +, ymws.frc1, ymws.frc2, U, lp)
        end
        
        Qslc .= reshape(Array(CUDA.reduce(+, ymws.rm; dims=tp)),lp.iL[end])./(32*pi^2)
    end    

    return sum(Qslc)
end
Qtop(U, lp::SpaceParm{4,M,D}, ymws::YMworkspace{T}) where {T,M,D} = Qtop(zeros(T,lp.iL[end],M), U, lp, ymws)


"""
    function Eoft_clover([Eslc,] U, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace)

Measure the action density `E(t)` using the clover discretization. If the argument `Eslc`
the contribution for each Euclidean time slice and plane are returned.
"""
function Eoft_clover(Eslc, U, lp::SpaceParm{4,M,B,D}, ymws::YMworkspace{T}) where {T,M,B,D}

    function acum(ipl1, ipl2, Etmp)

        tp = (1,2,3)
        V3 = prod(lp.iL[1:end-1])

        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_et!(ymws.rm, +, ymws.frc1, U, lp)
        end
        Etmp .=  reshape(Array(CUDA.reduce(+, ymws.rm;dims=tp)),lp.iL[end])/V3 
        for it in 1:lp.iL[end]
            Eslc[it,ipl1] = Etmp[it]/8
        end
        
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_et!(ymws.rm, +, ymws.frc2, U, lp)
        end
        Etmp .=  reshape(Array(CUDA.reduce(+, ymws.rm;dims=tp)),lp.iL[end])/V3 
        for it in 1:lp.iL[end]
            Eslc[it,ipl2] = Etmp[it]/8
        end

        return nothing
    end

    
    @timeit "E(t) clover measurement" begin

        fill!(Eslc,zero(T))
        Etmp = zeros(T,lp.iL[end])
        
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor!(ymws.frc1, ymws.frc2, U, 1,2, lp)
        end
        acum(1,2,Etmp)
        
        
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor!(ymws.frc1, ymws.frc2, U, 3,4, lp)
        end
        acum(3,4,Etmp)
        
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor!(ymws.frc1, ymws.frc2, U, 5,6, lp)
        end
        acum(5,6,Etmp)
        
    end    

    return sum(Eslc)/lp.iL[end]
end
Eoft_clover(U, lp::SpaceParm{N,M,B,D}, ymws::YMworkspace{T}) where {T,N,M,B,D} = Eoft_clover(zeros(T,lp.iL[end],M), U, lp, ymws)

function krnl_add_et!(rm, op, frc1, U, lp::SpaceParm{4,M,B,D}) where {M,B,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    X1 = (frc1[b,1,r]+frc1[b,2,r]+frc1[b,3,r]+frc1[b,4,r])
    
    I = point_coord((b,r), lp)
    rm[I] = dot(X1,X1)

    return nothing
end

function krnl_add_qd!(rm, op, frc1, frc2, U, lp::SpaceParm{4,M,B,D}) where {M,B,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    I = point_coord((b,r), lp)
    rm[I] = op(dot( (frc1[b,1,r]+frc1[b,2,r]+frc1[b,3,r]+frc1[b,4,r]),
                    (frc2[b,1,r]+frc2[b,2,r]+frc2[b,3,r]+frc2[b,4,r]) ) )
    return nothing
end

function krnl_field_tensor!(frc1, frc2, U::AbstractArray{T}, ipl1, ipl2, lp::SpaceParm{4,M,B,D}) where {T,M,B,D}

    b, r = CUDA.threadIdx().x, CUDA.blockIdx().x

    Ush = @cuStaticSharedMem(T, (D,2))

    #First plane
    id1, id2 = lp.plidx[ipl1]
    Ush[b,1] = U[b,id1,r]
    Ush[b,2] = U[b,id2,r]
    sync_threads()

    bu1, ru1 = up((b, r), id1, lp)
    bu2, ru2 = up((b, r), id2, lp)
    bd, rd   = up((bu1, ru1), id2, lp)
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
    
    l1 = gt1/gt2
    l2 = Ush[b,2]\Ush[b,1]

    frc1[b,1,r]     = projalg(Ush[b,1]*l1/Ush[b,2])
    frc1[bu1,2,ru1] = projalg(l1*l2)
    frc1[bd,3,rd]   = projalg(gt2\(l2*gt1))
    frc1[bu2,4,ru2] = projalg(l2*l1)

    # Second plane
    id1, id2 = lp.plidx[ipl2]
    Ush[b,1] = U[b,id1,r]
    Ush[b,2] = U[b,id2,r]
    sync_threads()

    bu1, ru1 = up((b, r), id1, lp)
    bu2, ru2 = up((b, r), id2, lp)
    bd, rd   = up((bu1, ru1), id2, lp)
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
    
    l1 = gt1/gt2
    l2 = Ush[b,2]\Ush[b,1]

    frc2[b,1,r]     = projalg(Ush[b,1]*l1/Ush[b,2])
    frc2[bu1,2,ru1] = projalg(l1*l2)
    frc2[bd,3,rd]   = projalg(gt2\(l2*gt1))
    frc2[bu2,4,ru2] = projalg(l2*l1)

    return nothing
end
    
