


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
        OBC = (B == BC_OPEN) && ((it == 1) || (it == lp.iL[end]))

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


        if !(SFBC && (it == 1)) && !OBC
            csw[b,ipl,r] = 0.125*(antsym(M1)+antsym(M2)+antsym(M3)+antsym(M4))
        end

    end

    return nothing
end



"""
    SF_bndfix!(sp, lp::Union{SpaceParm{4,6,BC_SF_ORBI,D},SpaceParm{4,6,BC_SF_AFWB,D}})

Sets all the values of `sp` in the  first time slice to zero.
"""
function SF_bndfix!(sp, lp::Union{SpaceParm{4,6,BC_SF_ORBI,D},SpaceParm{4,6,BC_SF_AFWB,D}}) where {D}
    @timeit "SF boundary fix" begin
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_sfbndfix!(sp, lp)
        end
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
    SF_bndfix!(sp, lp::SpaceParm{4,6,BC_OPEN,D})

Sets all the values of `sp` in the  first and last time slice to zero.
"""
function SF_bndfix!(sp, lp::SpaceParm{4,6,BC_OPEN,D}) where {D}
    @timeit "SF boundary fix" begin
        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_opbndfix!(sp, lp)
        end
    end
    return nothing
end

function krnl_opbndfix!(sp,lp::SpaceParm)
    b=Int64(CUDA.threadIdx().x)
    r=Int64(CUDA.blockIdx().x)

    if ((point_time((b,r),lp) == 1) || (point_time((b,r),lp) == lp.iL[end]))
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
