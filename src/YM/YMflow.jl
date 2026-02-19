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
        struct FlowIntr{N,T}

Structure containing info about a particular flow integrator
"""
struct FlowIntr{N,T}
    r::T
    e0::NTuple{N,T}
    e1::NTuple{N,T}

    add_zth::Bool
    c0::T
    
    eps::T
    tol::T
    eps_ini::T
    max_eps::T
    sft_fac::T
end

# pre-defined integrators
"""
        wfl_euler(::Type{T}, eps::T, tol::T)

Euler scheme integrator for the Wilson Flow. The fixed step size is given by `eps` and the tolerance for the adaptive integrators by `tol`. 
"""
wfl_euler(::Type{T}, eps::T, tol::T) where T = FlowIntr{0,T}(one(T),(),(),false,one(T),eps,tol,one(T)/200,one(T)/10,9/10)

"""
        zfl_euler(::Type{T}, eps::T, tol::T)

Euler scheme integrator for the Zeuthen flow. The fixed step size is given by `eps` and the tolerance for the adaptive integrators by `tol`. 
"""
zfl_euler(::Type{T}, eps::T, tol::T) where T = FlowIntr{0,T}(one(T),(),(),true, (one(T)*5)/3,eps,tol,one(T)/200,one(T)/10,9/10)

"""
        wfl_rk2(::Type{T}, eps::T, tol::T)

Second order Runge-Kutta integrator for the Wilson flow. The fixed step size is given by `eps` and the tolerance for the adaptive integrators by `tol`. 
"""
wfl_rk2(::Type{T}, eps::T, tol::T)   where T = FlowIntr{1,T}(one(T)/2,(-one(T)/2,),(one(T),),false,one(T),eps,tol,one(T)/200,one(T)/10,9/10)

"""
        zfl_rk2(::Type{T}, eps::T, tol::T)

Second order Runge-Kutta integrator for the Zeuthen flow. The fixed step size is given by `eps` and the tolerance for the adaptive integrators by `tol`. 
"""
zfl_rk2(::Type{T}, eps::T, tol::T)   where T = FlowIntr{1,T}(one(T)/2,(-one(T)/2,),(one(T),),true, (one(T)*5)/3,eps,tol,one(T)/200,one(T)/10,9/10)

"""
        wfl_rk3(::Type{T}, eps::T, tol::T)

Third order Runge-Kutta integrator for the Wilson flow. The fixed step size is given by `eps` and the tolerance for the adaptive integrators by `tol`. 
"""
wfl_rk3(::Type{T}, eps::T, tol::T)   where T = FlowIntr{2,T}(one(T)/4,(-17/36,-one(T)),(8/9,3/4),false,one(T),eps,tol,one(T)/200,one(T)/10,9/10)

"""
        Zfl_rk3(::Type{T}, eps::T, tol::T)

Third order Runge-Kutta integrator for the Zeuthen flow. The fixed step size is given by `eps` and the tolerance for the adaptive integrators by `tol`. 
"""
zfl_rk3(::Type{T}, eps::T, tol::T)   where T = FlowIntr{2,T}(one(T)/4,(-17/36,-one(T)),(8/9,3/4),true, (one(T)*5)/3,eps,tol,one(T)/200,one(T)/10,9/10)

function Base.show(io::IO, int::FlowIntr{N,T}) where {N,T}

    if (abs(int.c0-1) < 1.0E-10)
        println(io, "WILSON flow integrator")
    elseif (abs(int.c0-5/3) < 1.0E-10) && int.add_zth
        println(io, "ZEUTHEN flow integrator")
    elseif (abs(int.c0-5/3) < 1.0E-10) && !int.add_zth
        println(io, "SYMANZIK flow integrator")
    else
        println(io, "CUSTOM flow integrator")
        if int.add_zth
            println(io, "  - ", int.c0, " (with zeuthen term)")
        else
            println(io, "  - ", int.c0)
        end
    end

    if N == 0
        println(io, " * Euler schem3")
    elseif N == 1
        println(io, " * One stage scheme. Coefficients")
        println(io, "    stg 1: ", int.e0[1], " ", int.e1[1])
    elseif N == 2
        println(io, " * Two stage scheme. Coefficients:")
        println(io, "    stg 1: ", int.e0[1], " ", int.e1[1])
        println(io, "    stg 2: ", int.e0[2], " ", int.e1[2])
    end

    println(io, " * Fixed step size parameters: eps = ", int.eps)
    println(io, " * Adaptive step size parameters: tol = ", int.tol)
    println(io, "    - max eps:      ", int.max_eps)
    println(io, "    - initial eps:  ", int.eps_ini)
    println(io, "    - safety scale: ", int.sft_fac)

    return nothing
end


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

    @inbounds begin 
        b = Int64(CUDA.threadIdx().x)
        r = Int64(CUDA.blockIdx().x)
        it = point_time((b, r), lp)

        SFBC = ((B == BC_SF_AFWB) || (B == BC_SF_ORBI) )
        OBC = (B == BC_OPEN)

        @inbounds for id in 1:N
            bu, ru = up((b,r), id, lp)
            bd, rd = dw((b,r), id, lp)
            
            X = frc[bu,id,ru]
            Y  = frc[bd,id,rd]
            Ud = U[bd,id,rd]

            if SFBC
                if (it > 1) && (it < lp.iL[end])
                    frc2[b,id,r] = (5/6)*frc[b,id,r] + (1/6)*(projalg(Ud\Y*Ud) +
                        projalg(U[b,id,r]*X/U[b,id,r]))
                elseif (it == lp.iL[end]) && (id < N)
                    frc2[b,id,r] = (5/6)*frc[b,id,r] + (1/6)*(projalg(Ud\Y*Ud) +
                        projalg(U[b,id,r]*X/U[b,id,r]))
                end
            end
            if OBC
                if (it > 1) && (it < lp.iL[end])
                    frc2[b,id,r] = (5/6)*frc[b,id,r] + (1/6)*(projalg(Ud\Y*Ud) +
                        projalg(U[b,id,r]*X/U[b,id,r]))
                elseif ((it == lp.iL[end]) || (it == 1))  && (id < N)
                    frc2[b,id,r] = (5/6)*frc[b,id,r] + (1/6)*(projalg(Ud\Y*Ud) +
                        projalg(U[b,id,r]*X/U[b,id,r]))
                end
            else
                frc2[b,id,r] = (5/6)*frc[b,id,r] + (1/6)*(projalg(Ud\Y*Ud) +
                    projalg(U[b,id,r]*X/U[b,id,r]))
            end
        end
    end
    return nothing
end

"""
        function flw(U, int::FlowIntr{NI,T}, ns::Int64, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace)

Integrates the flow equations with the integration scheme defined by `int` performing `ns` steps with fixed step size. The configuration `U` is overwritten.   
"""
function flw(U, int::FlowIntr{NI,T}, ns::Int64, eps, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace) where {NI,T}    
    @timeit "Integrating flow equations" begin
        for i in 1:ns
            force_gauge(ymws, U, int.c0, 1, gp, lp)
            if int.add_zth
                add_zth_term(ymws::YMworkspace, U, lp)
            end
            ymws.mom .= ymws.frc1
            U .= expm.(U, ymws.mom, 2*eps*int.r)

            for k in 1:NI
                force_gauge(ymws, U, int.c0, 1, gp, lp)
                if int.add_zth
                    add_zth_term(ymws::YMworkspace, U, lp)
                end
                ymws.mom .= int.e0[k].*ymws.mom .+ int.e1[k].*ymws.frc1
                U .= expm.(U, ymws.mom, 2*eps)
            end
        end
    end
    
    return nothing
end
flw(U, int::FlowIntr{NI,T}, ns::Int64, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace) where {NI,T} = flw(U, int, ns, int.eps, gp, lp, ymws)


##
# Adaptive step size integrators
##

"""
        function flw_adapt(U, int::FlowIntr{NI,T}, tend::T, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace)

Integrates the flow equations with the integration scheme defined by `int` using the adaptive step size integrator up to `tend` with the tolerance defined in `int`. The configuration `U` is overwritten.   
"""
function flw_adapt(U, int::FlowIntr{NI,T}, tend::T, epsini::T, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace) where {NI,T}

    eps = epsini
    dt  = tend
    nstp = 0
    eps_all = Vector{T}(undef,0)
    while true
        ns = convert(Int64, floor(dt/eps))
        if ns > 10
            flw(U, int, 9, eps, gp, lp, ymws)
            ymws.U1 .= U
            flw(U, int, 1, eps, gp, lp, ymws)
            flw(ymws.U1, int, 2, eps/2, gp, lp, ymws)
            
            dt = dt - 10*eps
            nstp = nstp + 10
            push!(eps_all,ntuple(i->eps,10)...)

            # adjust step size
            ymws.U1 .= ymws.U1 ./ U
            maxd = CUDA.mapreduce(dev_one, max, ymws.U1, init=zero(tend))
            eps  = min(int.max_eps, 2*eps, int.sft_fac*eps*(int.tol/maxd)^(one(tend)/3))
            
        else
            flw(U, int, ns, eps, gp, lp, ymws)
            dt = dt - ns*eps

            push!(eps_all,ntuple(i->eps,ns)...)
            push!(eps_all,dt)

            flw(U, int, 1, dt, gp, lp, ymws)
            dt = zero(tend)

            nstp = nstp + ns + 1
        end
        
        if dt == zero(tend)
            break
        end
    end

    return nstp, eps_all
end
flw_adapt(U, int::FlowIntr{NI,T}, tend::T, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace) where {NI,T} = flw_adapt(U, int, tend, int.eps_ini, gp, lp, ymws)


##
# Observables
##


"""
    function Eoft_plaq([Eslc,] U, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace)

Measure the action density `E(t)` using the plaquette discretization. If the argument `Eslc` is given
the contribution for each Euclidean time slice and plane are returned.
"""
function Eoft_plaq(Eslc, U, gp::GaugeParm{T,G,NN}, lp::SpaceParm{N,M,B,D}, ymws::YMworkspace) where {T,G,NN,N,M,B,D}

    @timeit "E(t) plaquette measurement" begin

        ztw = ztwist(gp, lp)
        SFBC = ((B == BC_SF_AFWB) || (B == BC_SF_ORBI) )
        OBC = (B == BC_OPEN)

        tp = ntuple(i->i, N-1)
        V3 = prod(lp.iL[1:end-1])
        
        fill!(Eslc,zero(T))
        Etmp = zeros(T,lp.iL[end])
        for ipl in 1:M
            fill!(Etmp, zero(T))
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_plaq_pln!(ymws.cm, U, gp.Ubnd, ztw[ipl], ipl, lp)
            end

            Etmp .=  (gp.ng .- reshape(Array(CUDA.mapreduce(real, +, ymws.cm;dims=tp)),lp.iL[end]) ./ V3 )
            if ipl < N
                for it in 2:lp.iL[end]
                    Eslc[it,ipl] = Etmp[it] + Etmp[it-1]
                end
                if !SFBC
                    Eslc[1,ipl] = Etmp[1] + Etmp[end]
                end
                if OBC ## Check normalization of timelike boundary plaquettes
                    Eslc[end,ipl] = Etmp[end-1]
                    Eslc[1,ipl] = Etmp[1]
                end
            else
                for it in 1:lp.iL[end]
                    Eslc[it,ipl] = 2*Etmp[it]
                end
            end
        end
        
    end


    return sum(Eslc)/lp.iL[end]
end

Eoft_plaq(U, gp::GaugeParm{T,G,NN}, lp::SpaceParm{N,M,B,D}, ymws::YMworkspace) where {T,G,NN,N,M,B,D} = Eoft_plaq(zeros(T,lp.iL[end],M), U, gp, lp, ymws)


function krnl_plaq_pln!(plx, U::AbstractArray{T}, Ubnd, ztw, ipl, lp::SpaceParm{N,M,B,D}) where {T,N,M,B,D}

    @inbounds begin
        b = Int64(CUDA.threadIdx().x)
        r = Int64(CUDA.blockIdx().x)
        I = point_coord((b,r), lp)

        id1, id2 = lp.plidx[ipl]
        SFBC = ((B == BC_SF_AFWB) || (B == BC_SF_ORBI)) && (id1 == N)
        TWP  = ((I[id1]==1)&&(I[id2]==1))

        bu1, ru1 = up((b, r), id1, lp)
        bu2, ru2 = up((b, r), id2, lp)

        if SFBC && (point_time((b,r),lp) == lp.iL[end])
            gt = Ubnd[id2]
        else
            gt = U[bu1,id2,ru1]
        end

        if TWP
            plx[I] = ztw*tr(U[b,id1,r]*gt / (U[b,id2,r]*U[bu2,id1,ru2]))
        else
            plx[I] = tr(U[b,id1,r]*gt / (U[b,id2,r]*U[bu2,id1,ru2]))
        end
    end
    return nothing
end

"""
    Qtop([Qslc,] U, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace)

Measure the topological charge `Q` of the configuration `U` using the clover definition of the field strength tensor. If the argument `Qslc` is present the contributions for each Euclidean time slice are returned. Only works in 4D.
"""
function Qtop(Qslc, U, gp::GaugeParm, lp::SpaceParm{4,M,B,D}, ymws::YMworkspace) where {M,B,D}

    @timeit "Qtop measurement" begin

        ztw = ztwist(gp, lp)
        tp = (1,2,3)
        
        fill!(ymws.rm, zero(eltype(ymws.rm)))
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor!(ymws.frc1, ymws.frc2, U, gp.Ubnd, 1,5, ztw[1], ztw[5], lp)
        end
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_qd!(ymws.rm, -, ymws.frc1, ymws.frc2, lp)
        end
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor!(ymws.frc1, ymws.frc2, U, gp.Ubnd, 2,4, ztw[2], ztw[4], lp)
        end
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_qd!(ymws.rm, +, ymws.frc1, ymws.frc2, lp)
        end
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor!(ymws.frc1, ymws.frc2, U, gp.Ubnd, 3,6, ztw[3], ztw[6], lp)
        end
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_qd!(ymws.rm, -, ymws.frc1, ymws.frc2, lp)
        end
        Qslc .= reshape(Array(CUDA.reduce(+, ymws.rm; dims=tp)),lp.iL[end])./(32*pi^2)
    end    

    return sum(Qslc)
end
Qtop(U, gp::GaugeParm, lp::SpaceParm{4,M,D}, ymws::YMworkspace{T}) where {T,M,D} = Qtop(zeros(T,lp.iL[end]), U, gp, lp, ymws)


"""
    function Eoft_clover([Eslc,] U, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace)

Measure the action density `E(t)` using the clover discretization. If the argument `Eslc` is given
the contribution for each Euclidean time slice and plane are returned.
"""
function Eoft_clover(Eslc, U, gp::GaugeParm, lp::SpaceParm{4,M,B,D}, ymws::YMworkspace{T}) where {T,M,B,D}

    function acum(ipl1, ipl2, Etmp)

        tp = (1,2,3)
        V3 = prod(lp.iL[1:end-1])

        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_et!(ymws.rm, ymws.frc1, lp)
        end
        Etmp .=  reshape(Array(CUDA.reduce(+, ymws.rm;dims=tp)),lp.iL[end])/V3 
        for it in 1:lp.iL[end]
            Eslc[it,ipl1] = Etmp[it]/8
        end
        
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_et!(ymws.rm, ymws.frc2, lp)
        end
        Etmp .=  reshape(Array(CUDA.reduce(+, ymws.rm;dims=tp)),lp.iL[end])/V3 
        for it in 1:lp.iL[end]
            Eslc[it,ipl2] = Etmp[it]/8
        end

        return nothing
    end

    
    @timeit "E(t) clover measurement" begin

        ztw = ztwist(gp, lp)
        fill!(Eslc,zero(T))
        Etmp = zeros(T,lp.iL[end])
        
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor!(ymws.frc1, ymws.frc2, U, gp.Ubnd, 1,2, ztw[1], ztw[2], lp)
        end
        acum(1,2,Etmp)
        
        
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor!(ymws.frc1, ymws.frc2, U, gp.Ubnd, 3,4, ztw[3], ztw[4], lp)
        end
        acum(3,4,Etmp)
        
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor!(ymws.frc1, ymws.frc2, U, gp.Ubnd, 5,6, ztw[5], ztw[6], lp)
        end
        acum(5,6,Etmp)
        
    end    

    return sum(Eslc)/lp.iL[end]
end
Eoft_clover(U, gp::GaugeParm, lp::SpaceParm{N,M,B,D}, ymws::YMworkspace{T}) where {T,N,M,B,D} = Eoft_clover(zeros(T,lp.iL[end],M), U, gp, lp, ymws)

function krnl_add_et!(rm, frc1, lp::SpaceParm{4,M,B,D}) where {M,B,D}

    @inbounds begin
        b = Int64(CUDA.threadIdx().x)
        r = Int64(CUDA.blockIdx().x)

        X1 = (frc1[b,1,r]+frc1[b,2,r]+frc1[b,3,r]+frc1[b,4,r])
        
        I = point_coord((b,r), lp)
        rm[I] = dot(X1,X1)
    end

    return nothing
end

function krnl_add_qd!(rm, op, frc1, frc2, lp::SpaceParm{4,M,B,D}) where {M,B,D}

    @inbounds begin
        b = Int64(CUDA.threadIdx().x)
        r = Int64(CUDA.blockIdx().x)

        I = point_coord((b,r), lp)
        rm[I] += op(dot( (frc1[b,1,r]+frc1[b,2,r]+frc1[b,3,r]+frc1[b,4,r]),
                         (frc2[b,1,r]+frc2[b,2,r]+frc2[b,3,r]+frc2[b,4,r]) ) )
    end

    return nothing
end

function krnl_field_tensor!(frc1::AbstractArray{TA}, frc2, U::AbstractArray{T}, Ubnd, ipl1, ipl2, ztw1, ztw2, lp::SpaceParm{4,M,B,D}) where {TA,T,M,B,D}

    @inbounds begin
        b = Int64(CUDA.threadIdx().x)
        r = Int64(CUDA.blockIdx().x)
        I = point_coord((b,r), lp)
        it = I[4]
        
        #First plane
        id1, id2 = lp.plidx[ipl1]
        SFBC = ((B == BC_SF_AFWB) || (B == BC_SF_ORBI) ) && (id1 == 4)
        OBC = ((B == BC_OPEN) && (id1 == 4))
        TWP  = ((I[id1]==1)&&(I[id2]==1))
        
        bu1, ru1 = up((b, r), id1, lp)
        bu2, ru2 = up((b, r), id2, lp)
        bd, rd   = up((bu1, ru1), id2, lp)
        if SFBC && (it == lp.iL[end])
            gt1 = Ubnd[id2]
        else
            gt1 = U[bu1,id2,ru1]
        end
        
        l1 = gt1/U[bu2,id1,ru2]
        l2 = U[b,id2,r]\U[b,id1,r]

        if SFBC && (it == lp.iL[end])
            frc1[b,1,r]     = projalg(U[b,id1,r]*l1/U[b,id2,r])
            frc1[bu1,2,ru1] = zero(TA)
            frc1[bd,3,rd]   = zero(TA)
            frc1[bu2,4,ru2] = projalg(l2*l1)
        elseif OBC && (it == lp.iL[end])
            frc1[b,1,r]     = projalg(U[b,id1,r]*l1/U[b,id2,r])
            frc1[bu1,2,ru1] = zero(TA)
            frc1[bd,3,rd]   = zero(TA)
            frc1[bu2,4,ru2] = projalg(l2*l1)
        else
            if TWP
                frc1[b,1,r]     = projalg(ztw1, U[b,id1,r]*l1/U[b,id2,r])
                frc1[bu1,2,ru1] = projalg(ztw1, l1*l2)
                frc1[bd,3,rd]   = projalg(ztw1, U[bu2,id1,ru2]\(l2*gt1))
                frc1[bu2,4,ru2] = projalg(ztw1, l2*l1)
            else
                frc1[b,1,r]     = projalg(U[b,id1,r]*l1/U[b,id2,r])
                frc1[bu1,2,ru1] = projalg(l1*l2)
                frc1[bd,3,rd]   = projalg(U[bu2,id1,ru2]\(l2*gt1))
                frc1[bu2,4,ru2] = projalg(l2*l1)
            end
        end
        
        # Second plane
        id1, id2 = lp.plidx[ipl2]
        SFBC = ((B == BC_SF_AFWB) || (B == BC_SF_ORBI) ) && (id1 == 4)
        OBC = ((B == BC_OPEN) && (id1 == 4))
        TWP  = ((I[id1]==1)&&(I[id2]==1))

        bu1, ru1 = up((b, r), id1, lp)
        bu2, ru2 = up((b, r), id2, lp)
        bd, rd   = up((bu1, ru1), id2, lp)
        if SFBC && (it == lp.iL[end])
            gt1 = Ubnd[id2]
        else
            gt1 = U[bu1,id2,ru1]
        end
        
        l1 = gt1/U[bu2,id1,ru2]
        l2 = U[b,id2,r]\U[b,id1,r]

        if SFBC && (it == lp.iL[end])
            frc2[b,1,r]     = projalg(U[b,id1,r]*l1/U[b,id2,r])
            frc2[bu1,2,ru1] = zero(TA)
            frc2[bd,3,rd]   = zero(TA)
            frc2[bu2,4,ru2] = projalg(l2*l1)
        elseif OBC && (it == lp.iL[end])
            frc1[b,1,r]     = projalg(U[b,id1,r]*l1/U[b,id2,r])
            frc1[bu1,2,ru1] = zero(TA)
            frc1[bd,3,rd]   = zero(TA)
            frc1[bu2,4,ru2] = projalg(l2*l1)
        else
            if TWP
                frc2[b,1,r]     = projalg(ztw2, U[b,id1,r]*l1/U[b,id2,r])
                frc2[bu1,2,ru1] = projalg(ztw2, l1*l2)
                frc2[bd,3,rd]   = projalg(ztw2, U[bu2,id1,ru2]\(l2*gt1))
                frc2[bu2,4,ru2] = projalg(ztw2, l2*l1)
            else
                frc2[b,1,r]     = projalg(U[b,id1,r]*l1/U[b,id2,r])
                frc2[bu1,2,ru1] = projalg(l1*l2)
                frc2[bd,3,rd]   = projalg(U[bu2,id1,ru2]\(l2*gt1))
                frc2[bu2,4,ru2] = projalg(l2*l1)
            end
        end
    end
    return nothing
end

"""
    Qtop_rect([Qslc,] U, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace)

Measure the topological charge `Q` of the configuration `U` using the rectangle definition of the field strength tensor. If the argument `Qslc` is present the contributions for each Euclidean time slice are returned. Only works in 4D.
NOTE: So far, it is only valid for Periodic BC. For general BC, a consensus on boundary rectangles should be taken into account
"""
function Qtop_rect(Qslc, U, gp::GaugeParm, lp::SpaceParm{4,M,B,D}, ymws::YMworkspace) where {M,B,D}

    @timeit "Qtop_rect measurement" begin

        ztw = ztwist(gp, lp)
        tp = (1,2,3)
        
        fill!(ymws.rm, zero(eltype(ymws.rm)))
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor_rect!(ymws.frc1, ymws.frc2, U, gp.Ubnd, 1,5, ztw[1], ztw[5], lp)
        end
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_qd!(ymws.rm, -, ymws.frc1, ymws.frc2, lp)
        end
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor_rect!(ymws.frc1, ymws.frc2, U, gp.Ubnd, 2,4, ztw[2], ztw[4], lp)
        end
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_qd!(ymws.rm, +, ymws.frc1, ymws.frc2, lp)
        end
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor_rect!(ymws.frc1, ymws.frc2, U, gp.Ubnd, 3,6, ztw[3], ztw[6], lp)
        end
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_qd!(ymws.rm, -, ymws.frc1, ymws.frc2, lp)
        end
        Qslc .= reshape(Array(CUDA.reduce(+, ymws.rm; dims=tp)),lp.iL[end])./(32*pi^2)
    end    

    return sum(Qslc)
end
Qtop_rect(U, gp::GaugeParm, lp::SpaceParm{4,M,D}, ymws::YMworkspace{T}) where {T,M,D} = Qtop(zeros(T,lp.iL[end]), U, gp, lp, ymws)



function krnl_field_tensor_rect!(frc1::AbstractArray{TA}, frc2, U::AbstractArray{T}, Ubnd, ipl1, ipl2, ztw1, ztw2, lp::SpaceParm{4,M,B,D}) where {TA,T,M,B,D}
    # Convention adopted for point:
    #    H --- G
    #    |     |
    #    F --- E --- D
    #    |     |     |
    #    A --- B --- C

    @inbounds begin
        bA = Int64(CUDA.threadIdx().x)
        rA = Int64(CUDA.blockIdx().x)

        # --------------------------- First plane ---------------------------
        id1, id2 = lp.plidx[ipl1]

        bB,rB = up((bA,rA), id1, lp)
        bC,rC = up((bB,rB), id1, lp)
        bD,rD = up((bC,rC), id2, lp)
        bE,rE = up((bB,rB), id2, lp)
        bF,rF = up((bA,rA), id2, lp)
        bG,rG = up((bE,rE), id2, lp)
        bH,rH = up((bF,rF), id2, lp)

        # Horizontal rectangle
        l1 = U[bC,id2,rC] / (U[bF,id1,rF] * U[bE,id1,rE]) # CDEF
        l2 = U[bA,id2,rA] \ U[bA,id1,rA] * U[bB,id1,rB]   # FABC

        frc1[bA,1,rA] = projalg(U[bA,id1,rA] * U[bB,id1,rB] * l1 / U[b,id2,r])
        frc1[bC,2,rC] = projalg(l1 * l2)
        frc1[bD,3,rD] = projalg((U[bF,id1,rF] * U[bE,id1,rE]) \ l2 * U[bC,id2,rC])
        frc1[bF,4,rF] = projalg(l2 * l1)


        # Vertical rectangle
        l1 = U[bB,id2,rB] * U[bE,id2,rE] / U[bH,id1,rH]    # BEGH
        l2 = (U[bA,id2,rA] * U[bF,id2,rF]) \ U[bA,id1,rA]  # EFAB

        frc1[bA,1,rA] += projalg(U[bA,id1,rA] * l1 / (U[bA,id2,rA] * U[bF,id2,rF]))
        frc1[bB,2,rB] += projalg(l1 * l2)
        frc1[bG,3,rG] += projalg(U[bH,id1,rH] \ l2 * U[bB,id2,rB] * U[bE,id2,rE])
        frc1[bH,4,rH] += projalg(l2 * l1)




        # --------------------------- Second plane ---------------------------
        id1, id2 = lp.plidx[ipl2]

        bB,rB = up((bA,rA), id1, lp)
        bC,rC = up((bB,rB), id1, lp)
        bD,rD = up((bC,rC), id2, lp)
        bE,rE = up((bB,rB), id2, lp)
        bF,rF = up((bA,rA), id2, lp)
        bG,rG = up((bE,rE), id2, lp)
        bH,rH = up((bF,rF), id2, lp)

        # Horizontal rectangle
        l1 = U[bC,id2,rC] / (U[bF,id1,rF] * U[bE,id1,rE]) # CDEF
        l2 = U[bA,id2,rA] \ U[bA,id1,rA] * U[bB,id1,rB]   # FABC

        frc2[bA,1,rA] = projalg(U[bA,id1,rA] * U[bB,id1,rB] * l1 / U[b,id2,r])
        frc2[bC,2,rC] = projalg(l1 * l2)
        frc2[bD,3,rD] = projalg((U[bF,id1,rF] * U[bE,id1,rE]) \ l2 * U[bC,id2,rC])
        frc2[bF,4,rF] = projalg(l2 * l1)


        # Vertical rectangle
        l1 = U[bB,id2,rB] * U[bE,id2,rE] / U[bH,id1,rH]    # BEGH
        l2 = (U[bA,id2,rA] * U[bF,id2,rF]) \ U[bA,id1,rA]  # EFAB

        frc2[bA,1,rA] += projalg(U[bA,id1,rA] * l1 / (U[bA,id2,rA] * U[bF,id2,rF]))
        frc2[bB,2,rB] += projalg(l1 * l2)
        frc2[bG,3,rG] += projalg(U[bH,id1,rH] \ l2 * U[bB,id2,rB] * U[bE,id2,rE])
        frc2[bH,4,rH] += projalg(l2 * l1)
    end

    return nothing
end



#### PUNTI CRITICI
# 1.) Factors 2 in the implementation. 
# clover -> 1/4 * Im(clv)             =>   q = 1/32/π² ϵ*Tr[C*C] 
# rect   -> 1/8 * Im(rect1 + rect2)   =>   q = 2/32/π² ϵ*Tr[C*C]

# 2.) rect_v SUMS to frc1 and frc2 THEN krnl_add_qd is called. Is it right????


































# """
#     Qtop_rect([Qslc,] U, gp::GaugeParm, lp::SpaceParm, ymws::YMworkspace)

# Measure the topological charge `Q` of the configuration `U` using the rectangle definition of the field strength tensor. If the argument `Qslc` is present the contributions for each Euclidean time slice are returned. Only works in 4D.
# NOTE: So far, it is only valid for Periodic BC. For general BC, a consensus on boundary rectangles should be taken into account
# """
# function Qtop_rect(Qslc, U, gp::GaugeParm, lp::SpaceParm{4,M,BC_PERIODIC,D}, ymws::YMworkspace) where {M,D}#B,D}

#     @timeit "Qtop_rect measurement" begin

#         ztw = ztwist(gp, lp)
#         tp = (1,2,3)

#         fill!(ymws.rm, zero(eltype(ymws.rm)))
#         CUDA.@sync begin
#             CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor_rect_h!(ymws.frc1, ymws.frc2, U, gp.Ubnd, 1,5, ztw[1], ztw[5], lp)
#         end
#         CUDA.@sync begin
#             CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor_rect_v!(ymws.frc1, ymws.frc2, U, gp.Ubnd, 1,5, ztw[1], ztw[5], lp)
#         end
#         CUDA.@sync begin
#             CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_qd!(ymws.rm, -, ymws.frc1, ymws.frc2, lp)
#         end
#         CUDA.@sync begin
#             CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor_rect_h!(ymws.frc1, ymws.frc2, U, gp.Ubnd, 2,4, ztw[2], ztw[4], lp)
#         end
#         CUDA.@sync begin
#             CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor_rect_v!(ymws.frc1, ymws.frc2, U, gp.Ubnd, 2,4, ztw[2], ztw[4], lp)
#         end
#         CUDA.@sync begin
#             CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_qd!(ymws.rm, +, ymws.frc1, ymws.frc2, lp)
#         end
#         CUDA.@sync begin
#             CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor_rect_h!(ymws.frc1, ymws.frc2, U, gp.Ubnd, 3,6, ztw[3], ztw[6], lp)
#         end
#         CUDA.@sync begin
#             CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_field_tensor_rect_v!(ymws.frc1, ymws.frc2, U, gp.Ubnd, 3,6, ztw[3], ztw[6], lp)
#         end
#         CUDA.@sync begin
#             CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_add_qd!(ymws.rm, -, ymws.frc1, ymws.frc2, lp)
#         end
#         Qslc .= reshape(Array(CUDA.reduce(+, ymws.rm; dims=tp)),lp.iL[end])./(32*pi^2)
#     end

#     return sum(Qslc)
# end
# Qtop_rect(U, gp::GaugeParm, lp::SpaceParm{4,M,BC_PERIODIC,D}, ymws::YMworkspace{T}) where {T,M,D} = Qtop_rect(zeros(T,lp.iL[end]), U, gp, lp, ymws)


# """
#     krnl_field_tensor_rect_h!(frc1, frc2, U, Ubnd, ipl1, ipl2, lp)

# GPU kernel computing rectangles. Periodic BC only (no SFBC/OBC/TWP branches, `Ubnd` is not used).

# For each lattice site `(b,r)``, computes the four cyclic  rotations of the horizontal 2×1 rectangle (long side along `id1`) in the planes `ipl1` and `ipl2`, and distributes them to the slots `[b,1,r]`, `[bu11,2,ru11]`, `[bu112,3,ru112]`,  `[bu2,4,ru2]` of `frc1` and `frc2`, respectively. Represented graphically, for each plane  `ipl1` and `ipl2`:
# ```
#     F --- E --- D
#     |     |     |
#     A --- B --- C
    
#     A[1] = A → B → C → D → E → F → A
#     C[2] = C → D → E → F → A → B → C
#     D[3] = D → E → F → A → B → C → D
#     F[4] = F → A → B → C → D → E → F
# ```
# """
# function krnl_field_tensor_rect_h!(frc1::AbstractArray{TA}, frc2, U::AbstractArray{T}, Ubnd, ipl1, ipl2, lp::SpaceParm{4,M,B,D}) where {TA,T,M,B,D}

#     # Convention adopted for point:
#     #    F --- E --- D
#     #    |     |     |
#     #    A --- B --- C

#     @inbounds begin
#         bA = Int64(CUDA.threadIdx().x)
#         rA = Int64(CUDA.blockIdx().x)

#         # --------------------------- First plane ---------------------------
#         id1, id2 = lp.plidx[ipl1]

#         bB,rB = up((bA,rA), id1, lp)
#         bC,rC = up((bB,rB), id1, lp)
#         bD,rD = up((bC,rC), id2, lp)
#         bE,rE = up((bB,rB), id2, lp)
#         bF,rF = up((bA,rA), id2, lp)

#         l1 = U[bC,id2,rC] / (U[bF,id1,rF] * U[bE,id1,rE]) # CDEF
#         l2 = U[bA,id2,rA] \ U[bA,id1,rA] * U[bB,id1,rB]   # FABC

#         frc1[bA,1,rA] = projalg(U[bA,id1,rA] * U[bB,id1,rB] * l1 / U[b,id2,r])
#         frc1[bC,2,rC] = projalg(l1 * l2)
#         frc1[bD,3,rD] = projalg((U[bF,id1,rF] * U[bE,id1,rE]) \ l2 * U[bC,id2,rC])
#         frc1[bF,4,rF] = projalg(l2 * l1)


#         # --------------------------- Second plane ---------------------------
#         id1, id2 = lp.plidx[ipl2]

#         bB,rB = up((bA,rA), id1, lp)
#         bC,rC = up((bB,rB), id1, lp)
#         bD,rD = up((bC,rC), id2, lp)
#         bE,rE = up((bB,rB), id2, lp)
#         bF,rF = up((bA,rA), id2, lp)

#         l1 = U[bC,id2,rC] / (U[bF,id1,rF] * U[bE,id1,rE]) # CDEF
#         l2 = U[bA,id2,rA] \ U[bA,id1,rA] * U[bB,id1,rB]   # FABC

#         frc2[bA,1,rA] = projalg(U[bA,id1,rA] * U[bB,id1,rB] * l1 / U[b,id2,r])
#         frc2[bC,2,rC] = projalg(l1 * l2)
#         frc2[bD,3,rD] = projalg((U[bF,id1,rF] * U[bE,id1,rE]) \ l2 * U[bC,id2,rC])
#         frc2[bF,4,rF] = projalg(l2 * l1)
#     end

#     return nothing
# end



# """
#     krnl_field_tensor_rect_v!(frc1, frc2, U, ipl1, ipl2, lp)

# GPU kernel computing rectangles. Periodic BC only (no SFBC/OBC/TWP branches, `Ubnd` is not used).

# For each lattice site `(b,r)`, computes the four cyclic rotations of the vertical 1×2 rectangle (long side along `id2`) in the planes `ipl1` and `ipl2`, and distributes them to the slots `[b,1,r]`, `[bu1,2,ru1]`, `[bu112,3,ru112]`,  `[bu2_u2,4,ru2_u2]` of `frc1` and `frc2`, respectively.  Periodic BC only (no SFBC/OBC/TWP branches). Represented graphically, for each plane `ipl1` and `ipl2`:
# ```
#     E -- D
#     |    |
#     F -- C
#     |    |
#     A -- B
# ```
# """
# function krnl_field_tensor_rect_v!(frc1::AbstractArray{TA}, frc2, U::AbstractArray{T}, Ubnd, ipl1, ipl2, lp::SpaceParm{4,M,B,D}) where {TA,T,M,B,D}

#     # Convention adopted for point:
#     #    E -- D
#     #    |    |
#     #    F -- C
#     #    |    |
#     #    A -- B

#     @inbounds begin
#         bA = Int64(CUDA.threadIdx().x)
#         rA = Int64(CUDA.blockIdx().x)

#         # --------------------------- First plane ---------------------------
#         id1, id2 = lp.plidx[ipl1]

#         bB,rB = up((bA,rA), id1, lp)
#         bC,rC = up((bB,rB), id2, lp)
#         bD,rD = up((bC,rC), id2, lp)
#         bF,rF = up((bA,rA), id2, lp)
#         bE,rE = up((bF,rF), id2, lp)

#         l1 = U[bB,id2,rB] * U[bC,id2,rC] / U[bE,id1,rE]    # BCDE
#         l2 = (U[bA,id2,rA] * U[bF,id2,rF]) \ U[bA,id1,rA]  # EFAB

#         frc1[bA,1,rA] += projalg(U[bA,id1,rA] * l1 / U[bE,id1,rE])
#         frc1[bB,2,rB] += projalg(l1 * l2)
#         frc1[bD,3,rD] += projalg(U[bE,id1,rE] \ l2 * U[bB,id2,rB] * U[bC,id2,rC])
#         frc1[bE,4,rE] += projalg(l2 * l1)


#         # --------------------------- Second plane ---------------------------
#         id1, id2 = lp.plidx[ipl2]

#         bB,rB = up((bA,rA), id1, lp)
#         bC,rC = up((bB,rB), id1, lp)
#         bD,rD = up((bC,rC), id2, lp)
#         bE,rE = up((bB,rB), id2, lp)
#         bF,rF = up((bA,rA), id2, lp)

#         l1 = U[bB,id2,rB] * U[bC,id2,rC] / U[bE,id1,rE]    # BCDE
#         l2 = (U[bA,id2,rA] * U[bF,id2,rF]) \ U[bA,id1,rA]  # EFAB

#         frc2[bA,1,rA] += projalg(U[bA,id1,rA] * l1 / U[bE,id1,rE])
#         frc2[bB,2,rB] += projalg(l1 * l2)
#         frc2[bD,3,rD] += projalg(U[bE,id1,rE] \ l2 * U[bB,id2,rB] * U[bC,id2,rC])
#         frc2[bE,4,rE] += projalg(l2 * l1)

#     end
#     return nothing
# end



