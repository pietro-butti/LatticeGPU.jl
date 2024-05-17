
import ..YM.flw, ..YM.force_gauge, ..YM.flw_adapt


function flw(U, psi, int::FlowIntr{NI,T}, ns::Int64, eps, gp::GaugeParm, dpar::DiracParam, lp::SpaceParm, ymws::YMworkspace, dws::DiracWorkspace) where {NI,T}
    @timeit "Integrating flow equations" begin
        for i in 1:ns
            force_gauge(ymws, U, int.c0, 1, gp, lp)

            if int.add_zth
                add_zth_term(ymws::YMworkspace, U, lp)
            end

            Nablanabla!(dws.sAp, U, psi, dpar, dws, lp)
            psi .= psi + 2*int.r*eps*dws.sAp

            ymws.mom .= ymws.frc1
            U .= expm.(U, ymws.mom, 2*eps*int.r)

            for k in 1:NI
                force_gauge(ymws, U, int.c0, 1, gp, lp)

                if int.add_zth
                    add_zth_term(ymws::YMworkspace, U, lp)
                end

                Nablanabla!(dws.sp, U, psi, dpar, dws, lp)
                dws.sAp .= int.e0[k].*dws.sAp .+ int.e1[k].*dws.sp
                psi .= psi + 2*eps*dws.sAp

                ymws.mom .= int.e0[k].*ymws.mom .+ int.e1[k].*ymws.frc1
                U .= expm.(U, ymws.mom, 2*eps)
           end
        end
    end

    return nothing
end
flw(U, psi, int::FlowIntr{NI,T}, ns::Int64, gp::GaugeParm, dpar::DiracParam, lp::SpaceParm, ymws::YMworkspace, dws::DiracWorkspace) where {NI,T} = flw(U, psi, int::FlowIntr{NI,T}, ns::Int64, int.eps, gp::GaugeParm, dpar::DiracParam, lp::SpaceParm, ymws::YMworkspace, dws::DiracWorkspace)

"""
    function backflow(psi, U, Dt, nsave::Int64, gp::GaugeParm, dpar::DiracParam, lp::SpaceParm, ymws::YMworkspace, dws::DiracWorkspace)

Performs one step back in flow time for the fermion field, according to 1302.5246. The fermion field must me that of the time-slice Dt and is flowed back to the first time-slice
nsave is the total number of gauge fields saved in the process

"""
function backflow(psi, U, Dt, maxnsave::Int64, gp::GaugeParm, dpar::DiracParam, lp::SpaceParm, ymws::YMworkspace, dws::DiracWorkspace)

    int = wfl_rk3(Float64,0.01,1.0) # Default integrator, it has to be order 3 rk but in can be zfl

    @timeit "Backflow integration" begin
        @timeit "GPU to CPU" U0 = Array(U)

        nt,eps_all = flw_adapt(U, int, Dt, gp, lp, ymws)

        nsave = min(maxnsave,nt)

        nsave != 0 ? dsave = Int64(floor(nt/nsave)) : dsave = nt
        Usave = Vector{typeof(U0)}(undef,nsave)

        @timeit "CPU to GPU" copyto!(U,U0)
        for i in 1:(dsave*nsave)
            flw(U, int, 1, eps_all[i], gp, lp, ymws)
            (i%dsave)==0 ? Usave[Int64(i/dsave)] = Array(U) : nothing
        end

        for j in (nt%nsave):-1:1
            @timeit "CPU to GPU" copyto!(U,Usave[end])
            for k in 1:j-1
                flw(U, int, 1, eps_all[nsave*dsave + k], gp, lp, ymws)
            end
            bflw_step!(psi, U, eps_all[nsave*dsave + j], int::FlowIntr, gp::GaugeParm, dpar::DiracParam, lp::SpaceParm, ymws::YMworkspace, dws::DiracWorkspace)
        end

        for i in (nsave-1):-1:1
            for j in dsave:-1:1
                @timeit "CPU to GPU" copyto!(U,Usave[i])
                for k in 1:j-1
                    flw(U, int, 1, eps_all[i*dsave + k], gp, lp, ymws)
                end
                bflw_step!(psi, U, eps_all[i*dsave + j], int::FlowIntr, gp::GaugeParm, dpar::DiracParam, lp::SpaceParm, ymws::YMworkspace, dws::DiracWorkspace)
            end
        end

        @timeit "CPU to GPU" copyto!(U,U0)

        for j in dsave:-1:1
        @timeit "CPU to GPU" copyto!(U,U0)
            for k in 1:j-1
                flw(U, int, 1, eps_all[k], gp, lp, ymws)
            end
            bflw_step!(psi, U, eps_all[j], int::FlowIntr, gp::GaugeParm, dpar::DiracParam, lp::SpaceParm, ymws::YMworkspace, dws::DiracWorkspace)
        end

        @timeit "CPU to GPU" copyto!(U,U0)
    end

    return nothing
end

"""
function bflw_step!(U, psi, eps, int::FlowIntr, gp::GaugeParm, dpar::DiracParam, lp::SpaceParm, ymws::YMworkspace, dws::DiracWorkspace)

Performs ONE backstep in psi, from t to t-\eps. U is supposed to be the one in t-\eps and is left unchanged. So far, int has to be rk4
"""
function bflw_step!(psi, U,  eps, int::FlowIntr, gp::GaugeParm, dpar::DiracParam, lp::SpaceParm, ymws::YMworkspace, dws::DiracWorkspace)

    @timeit "Backflow step" begin

        V = copy(U)
        V .= U

        force_gauge(ymws, U, int.c0, 1, gp, lp)

        if int.add_zth
            add_zth_term(ymws::YMworkspace, U, lp)
        end

        ymws.mom .= ymws.frc1
        U .= expm.(U, ymws.mom, 2*eps*int.r)

        force_gauge(ymws, U, int.c0, 1, gp, lp)

        if int.add_zth
            add_zth_term(ymws::YMworkspace, U, lp)
        end

        ymws.mom .= int.e0[1].*ymws.mom .+ int.e1[1].*ymws.frc1
        U .= expm.(U, ymws.mom, 2*eps)

        Nablanabla!(dws.sp, U, 0.75*2*eps*psi, dpar, dws, lp)

        U .= V

        force_gauge(ymws, U, int.c0, 1, gp, lp)

        if int.add_zth
            add_zth_term(ymws::YMworkspace, U, lp)
        end

        U .= expm.(U, ymws.frc1, 2*eps*int.r)

        Nablanabla!(dws.sAp, U, 2*eps*dws.sp, dpar, dws, lp)
        dws.sAp .= psi + (8/9)*dws.sAp

        U .= V

        Nablanabla!(psi, U, 2*eps*(dws.sAp - (8/9)*dws.sp), dpar, dws, lp)
        psi .= (1/4)*psi + dws.sp + dws.sAp

    end

    return nothing
end


function flw_adapt(U, psi, int::FlowIntr{NI,T}, tend::T, epsini::T, gp::GaugeParm, dpar::DiracParam, lp::SpaceParm, ymws::YMworkspace, dws::DiracWorkspace) where {NI,T}

    eps = epsini
    dt  = tend
    nstp = 0
    eps_all = Vector{T}(undef,0)
    while true
        ns = convert(Int64, floor(dt/eps))
        if ns > 10
            flw(U, psi, int, 9, eps, gp, dpar, lp, ymws, dws)
            ymws.U1 .= U
            flw(U, psi, int, 1, eps, gp, dpar, lp, ymws, dws)
            flw(ymws.U1, int, 2, eps/2, gp, lp, ymws)

            dt = dt - 10*eps
            nstp = nstp + 10
            push!(eps_all,ntuple(i->eps,10)...)

            # adjust step size
            ymws.U1 .= ymws.U1 ./ U
            maxd = CUDA.mapreduce(dev_one, max, ymws.U1, init=zero(tend))
            eps  = min(int.max_eps, 2*eps, int.sft_fac*eps*(int.tol/maxd)^(one(tend)/3))

        else
            flw(U, psi, int, ns, eps, gp, dpar, lp, ymws, dws)
            dt = dt - ns*eps

            push!(eps_all,ntuple(i->eps,ns)...)
            push!(eps_all,dt)

            flw(U, psi, int, 1, dt, gp, dpar, lp, ymws, dws)
            dt = zero(tend)

            nstp = nstp + ns + 1
        end

        if dt == zero(tend)
            break
        end
    end

    return nstp, eps_all
end
flw_adapt(U, psi, int::FlowIntr{NI,T}, tend::T, gp::GaugeParm, dpar::DiracParam, lp::SpaceParm, ymws::YMworkspace, dws::DiracWorkspace) where {NI,T} = flw_adapt(U, psi, int, tend, int.eps_ini, gp, dpar, lp, ymws, dws)


"""

    function Nablanabla!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::SpaceParm{4,6,B,D})

Computes /`/` \\nabla^* \\nabla /`/` `si` and stores it in `si`.

"""
function Nablanabla!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::SpaceParm{4,6,BC_PERIODIC,D}) where {D}
        @timeit "Laplacian" begin
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_Nablanabla(so, U, si, dpar.th, lp)
            end
        end
    return nothing
end
function Nablanabla!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::Union{SpaceParm{4,6,BC_SF_ORBI,D},SpaceParm{4,6,BC_SF_AFWB,D},SpaceParm{4,6,BC_OPEN,D}}) where {D}
    SF_bndfix!(si,lp)
        @timeit "Laplacian" begin
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_Nablanabla(so, U, si, dpar.th, lp)
            end
        end
    SF_bndfix!(so,lp)
    return nothing
end


function krnl_Nablanabla(so, U, si, th, lp::SpaceParm{4,6,BC_OPEN,D}) where {D}

    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    @inbounds begin

        if ((point_time((b,r),lp) != 1) && (point_time((b,r),lp) != lp.iL[end])

        so[b,r] = -4*si[b,r]

	        bu1, ru1 = up((b,r), 1, lp)
            bd1, rd1 = dw((b,r), 1, lp)
            bu2, ru2 = up((b,r), 2, lp)
            bd2, rd2 = dw((b,r), 2, lp)
            bu3, ru3 = up((b,r), 3, lp)
            bd3, rd3 = dw((b,r), 3, lp)
            bu4, ru4 = up((b,r), 4, lp)
            bd4, rd4 = dw((b,r), 4, lp)

        so[b,r] += 0.5*( th[1] * (U[b,1,r]*si[bu1,ru1]) +conj(th[1]) * (U[bd1,1,rd1]\si[bd1,rd1]) +
                         th[2] * (U[b,2,r]*si[bu2,ru2]) +conj(th[2]) * (U[bd2,2,rd2]\si[bd2,rd2]) +
                         th[3] * (U[b,3,r]*si[bu3,ru3]) +conj(th[3]) * (U[bd3,3,rd3]\si[bd3,rd3]) +
                         th[4] * (U[b,4,r]*si[bu4,ru4]) +conj(th[4]) * (U[bd4,4,rd4]\si[bd4,rd4])  )
        end
    end

    return nothing
end

function krnl_Nablanabla(so, U, si, th, lp::SpaceParm{4,6,BC_PERIODIC,D}) where {D}

    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    @inbounds begin

        so[b,r] = -4*si[b,r]

	        bu1, ru1 = up((b,r), 1, lp)
            bd1, rd1 = dw((b,r), 1, lp)
            bu2, ru2 = up((b,r), 2, lp)
            bd2, rd2 = dw((b,r), 2, lp)
            bu3, ru3 = up((b,r), 3, lp)
            bd3, rd3 = dw((b,r), 3, lp)
            bu4, ru4 = up((b,r), 4, lp)
            bd4, rd4 = dw((b,r), 4, lp)

        so[b,r] += 0.5*( th[1] * (U[b,1,r]*si[bu1,ru1]) +conj(th[1]) * (U[bd1,1,rd1]\si[bd1,rd1]) +
                         th[2] * (U[b,2,r]*si[bu2,ru2]) +conj(th[2]) * (U[bd2,2,rd2]\si[bd2,rd2]) +
                         th[3] * (U[b,3,r]*si[bu3,ru3]) +conj(th[3]) * (U[bd3,3,rd3]\si[bd3,rd3]) +
                         th[4] * (U[b,4,r]*si[bu4,ru4]) +conj(th[4]) * (U[bd4,4,rd4]\si[bd4,rd4])  )
    end

    return nothing
end

function krnl_Nablanabla(so, U, si, th, lp::Union{SpaceParm{4,6,BC_SF_ORBI,D},SpaceParm{4,6,BC_SF_AFWB,D}}) where {D}

    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    @inbounds begin

        if (point_time((b,r),lp) != 1)

        so[b,r] = -4*si[b,r]

	        bu1, ru1 = up((b,r), 1, lp)
            bd1, rd1 = dw((b,r), 1, lp)
            bu2, ru2 = up((b,r), 2, lp)
            bd2, rd2 = dw((b,r), 2, lp)
            bu3, ru3 = up((b,r), 3, lp)
            bd3, rd3 = dw((b,r), 3, lp)
            bu4, ru4 = up((b,r), 4, lp)
            bd4, rd4 = dw((b,r), 4, lp)

        so[b,r] += 0.5*( th[1] * (U[b,1,r]*si[bu1,ru1]) +conj(th[1]) * (U[bd1,1,rd1]\si[bd1,rd1]) +
                         th[2] * (U[b,2,r]*si[bu2,ru2]) +conj(th[2]) * (U[bd2,2,rd2]\si[bd2,rd2]) +
                         th[3] * (U[b,3,r]*si[bu3,ru3]) +conj(th[3]) * (U[bd3,3,rd3]\si[bd3,rd3]) +
                         th[4] * (U[b,4,r]*si[bu4,ru4]) +conj(th[4]) * (U[bd4,4,rd4]\si[bd4,rd4])  )
        end
    end

    return nothing
end



export Nablanabla!, flw, backflow, flw_adapt, bflw_step!


"""
    function Dslash_sq!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::SpaceParm{4,6,B,D})

Computes /`/` //slashed{D}^2 si /`/` ans stores it in `si`.

"""
function Dslash_sq!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::SpaceParm{4,6,B,D}) where {B,D}

        @timeit "DwdagDw" begin

            @timeit "g5Dslsh" begin
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dslsh!(dws.st, U, si, dpar.th, lp)
            end
            end

            if abs(dpar.csw) > 1.0E-10
                @timeit "Dw_improvement" begin
                        CUDA.@sync begin
                            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dslsh_impr!(dws.st, dws.csw, dpar.csw, si,  lp)
                        end
                end
            end


            @timeit "g5Dslsh" begin
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dslsh!(so, U, dws.st, dpar.th, lp)
            end
            end

            if abs(dpar.csw) > 1.0E-10
                @timeit "Dw_improvement" begin
                        CUDA.@sync begin
                            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dslsh_impr!(so, dws.csw, dpar.csw, dws.st,  lp)
                        end
                end
            end


        end

    return nothing
end


function krnl_g5Dslsh!(so, U, si, th, lp::Union{SpaceParm{4,6,BC_SF_ORBI,D},SpaceParm{4,6,BC_SF_AFWB,D}}) where {D}

    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    if (point_time((b,r),lp) != 1)

        @inbounds begin

            so[b,r] = 4*si[b,r]

            bu1, ru1 = up((b,r), 1, lp)
            bd1, rd1 = dw((b,r), 1, lp)
            bu2, ru2 = up((b,r), 2, lp)
            bd2, rd2 = dw((b,r), 2, lp)
            bu3, ru3 = up((b,r), 3, lp)
            bd3, rd3 = dw((b,r), 3, lp)
            bu4, ru4 = up((b,r), 4, lp)
            bd4, rd4 = dw((b,r), 4, lp)

       so[b,r] -= 0.5*(th[1]*gpmul(Pgamma{1,-1},U[b,1,r],si[bu1,ru1]) +conj(th[1])*gdagpmul(Pgamma{1,+1},U[bd1,1,rd1],si[bd1,rd1]) +
                        th[2]*gpmul(Pgamma{2,-1},U[b,2,r],si[bu2,ru2]) +conj(th[2])*gdagpmul(Pgamma{2,+1},U[bd2,2,rd2],si[bd2,rd2]) +
                        th[3]*gpmul(Pgamma{3,-1},U[b,3,r],si[bu3,ru3]) +conj(th[3])*gdagpmul(Pgamma{3,+1},U[bd3,3,rd3],si[bd3,rd3]) +
                        th[4]*gpmul(Pgamma{4,-1},U[b,4,r],si[bu4,ru4]) +conj(th[4])*gdagpmul(Pgamma{4,+1},U[bd4,4,rd4],si[bd4,rd4]) )

        so[b,r] = dmul(Gamma{5}, so[b,r])
        end
    end
    return nothing
end


function krnl_g5Dslsh!(so, U, si, th, lp::SpaceParm{4,6,B,D}) where {D,B}

    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    @inbounds begin

        so[b,r] = 4*si[b,r]

	        bu1, ru1 = up((b,r), 1, lp)
            bd1, rd1 = dw((b,r), 1, lp)
            bu2, ru2 = up((b,r), 2, lp)
            bd2, rd2 = dw((b,r), 2, lp)
            bu3, ru3 = up((b,r), 3, lp)
            bd3, rd3 = dw((b,r), 3, lp)
            bu4, ru4 = up((b,r), 4, lp)
            bd4, rd4 = dw((b,r), 4, lp)

       so[b,r] -= 0.5*(th[1]*gpmul(Pgamma{1,-1},U[b,1,r],si[bu1,ru1]) +conj(th[1])*gdagpmul(Pgamma{1,+1},U[bd1,1,rd1],si[bd1,rd1]) +
                        th[2]*gpmul(Pgamma{2,-1},U[b,2,r],si[bu2,ru2]) +conj(th[2])*gdagpmul(Pgamma{2,+1},U[bd2,2,rd2],si[bd2,rd2]) +
                        th[3]*gpmul(Pgamma{3,-1},U[b,3,r],si[bu3,ru3]) +conj(th[3])*gdagpmul(Pgamma{3,+1},U[bd3,3,rd3],si[bd3,rd3]) +
                        th[4]*gpmul(Pgamma{4,-1},U[b,4,r],si[bu4,ru4]) +conj(th[4])*gdagpmul(Pgamma{4,+1},U[bd4,4,rd4],si[bd4,rd4]) )

        so[b,r] = dmul(Gamma{5}, so[b,r])
    end

    return nothing
end

function krnl_g5Dslsh_impr!(so, Fcsw, csw, si, lp::SpaceParm{4,6,B,D}) where {B,D}

    @inbounds begin

    b = Int64(CUDA.threadIdx().x);
    r = Int64(CUDA.blockIdx().x)

        so[b,r] += 0.5*csw*im*dmul(Gamma{5},( Fcsw[b,1,r]*dmul(Gamma{10},si[b,r]) + Fcsw[b,2,r]*dmul(Gamma{11},si[b,r]) + Fcsw[b,3,r]*dmul(Gamma{12},si[b,r])
                                          -Fcsw[b,4,r]*dmul(Gamma{15},si[b,r]) - Fcsw[b,5,r]*dmul(Gamma{14},si[b,r]) - Fcsw[b,6,r]*dmul(Gamma{13},si[b,r])))
    end

    return nothing
end



function krnl_g5Dslsh_impr!(so, Fcsw, csw, si, lp::Union{SpaceParm{4,6,BC_SF_ORBI,D},SpaceParm{4,6,BC_SF_AFWB,D}}) where {D}

    @inbounds begin

    b = Int64(CUDA.threadIdx().x);
    r = Int64(CUDA.blockIdx().x)

    if (point_time((b,r),lp) != 1)

        so[b,r] += 0.5*csw*im*dmul(Gamma{5},( Fcsw[b,1,r]*dmul(Gamma{10},si[b,r]) + Fcsw[b,2,r]*dmul(Gamma{11},si[b,r]) + Fcsw[b,3,r]*dmul(Gamma{12},si[b,r])
                                          -Fcsw[b,4,r]*dmul(Gamma{15},si[b,r]) - Fcsw[b,5,r]*dmul(Gamma{14},si[b,r]) - Fcsw[b,6,r]*dmul(Gamma{13},si[b,r])))
    end

    return nothing
    end
end
