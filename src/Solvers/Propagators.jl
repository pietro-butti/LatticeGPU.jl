###
### "THE BEER-WARE LICENSE":
### Fernando Panadero wrote this file based on Alberto Ramos' work.
### As long as you retain this notice you can do whatever you want with this stuff. If we meet some
### day, and you think this stuff is worth it, you can buy us a beer in
### return. <alberto.ramos@cern.ch> <fernando.p@csic.es>
###
### file:    Propagators.jl
###


"""
    function propagator!(pro,U, dpar::DiracParam{T}, dws::DiracWorkspace,  lp::SpaceParm, maxiter::Int64, tol::Float64, y::NTuple{4,Int64}, c::Int64, s::Int64)

Saves the fermionic progapator in pro for a source at point `y` with color `c` and spin `s`. If the last three arguments are replaced by `time::Int64`, the source is replaced
by a random source in spin and color at t = `time`. Returns the number of iterations.

"""
function propagator!(pro, U, dpar::DiracParam{T}, dws::DiracWorkspace, lp::SpaceParm, maxiter::Int64, tol::Float64, y::NTuple{4,Int64}, c::Int64, s::Int64) where {T}

    function krnlg5!(src)
        b=Int64(CUDA.threadIdx().x)
        r=Int64(CUDA.blockIdx().x)
        src[b,r] = dmul(Gamma{5},src[b,r])
        return nothing
    end

    @timeit "Propagator computation" begin

        fill!(dws.sp,zero(eltype(scalar_field(Spinor{4,SU3fund{Float64}},lp))))

        CUDA.@allowscalar dws.sp[point_index(CartesianIndex{lp.ndim}(y),lp)...] = Spinor{4,SU3fund{Float64}}(ntuple(i -> (i==s)*SU3fund{Float64}(ntuple(j -> (j==c)*1.0,3)...),4))

        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnlg5!(dws.sp)
        end

        g5Dw!(pro,U,dws.sp,mtwmdpar(dpar),dws,lp)

        niter = CG!(pro,U,DwdagDw!,dpar,lp,dws,maxiter,tol)
    end

    return niter
end

function propagator!(pro, U, dpar::DiracParam{T}, dws::DiracWorkspace, lp::SpaceParm, maxiter::Int64, tol::Float64, time::Int64, psi0 = nothing) where {T}

    function krnlg5!(src)
        b=Int64(CUDA.threadIdx().x)
        r=Int64(CUDA.blockIdx().x)
        src[b,r] = dmul(Gamma{5},src[b,r])
        return nothing
    end

    @timeit "Propagator computation" begin
        fill!(dws.sp,zero(eltype(scalar_field(Spinor{4,SU3fund{Float64}},lp))))

        pfrandomize!(dws.sp,lp,time)

        if !isnothing(psi0)
            psi0 .= Array(dws.sp)
        end

        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnlg5!(dws.sp)
        end

        g5Dw!(pro,U,dws.sp,mtwmdpar(dpar),dws,lp)

        niter = CG!(pro,U,DwdagDw!,dpar,lp,dws,maxiter,tol)
    end

    return niter
end

"""

    function bndpropagator!(pro,U, dpar::DiracParam{T}, dws::DiracWorkspace, lp::SpaceParm{4,6,1,D}, maxiter::Int64, tol::Float64, c::Int64, s::Int64)
        
Saves the propagator from the t=0 boundary to the bulk for the SF boundary conditions for a source with color 'c' and spin 's' in 'pro'. The factor c_t is included while the factor 1/sqrt(V) is not.
For the propagator from T to the bulk, use the function Tbndpropagator(U, dpar::DiracParam{T}, dws::DiracWorkspace, lp::SpaceParm{4,6,1,D}, maxiter::Int64, tol::Float64, c::Int64, s::Int64). Returns the number of iterations.

"""
function bndpropagator!(pro, U, dpar::DiracParam{T}, dws::DiracWorkspace, lp::SpaceParm{4,6,1,D}, maxiter::Int64, tol::Float64, c::Int64, s::Int64) where {T,D}

    function krnlg5!(src)
        b=Int64(CUDA.threadIdx().x)
        r=Int64(CUDA.blockIdx().x)
        src[b,r] = dmul(Gamma{5},src[b,r])
        return nothing
    end

    function krnl_assign_bndsrc!(src,U,lp::SpaceParm, c::Int64, s::Int64)
        b=Int64(CUDA.threadIdx().x)
        r=Int64(CUDA.blockIdx().x)

        if (point_time((b,r),lp) == 2)
            bd4, rd4 = dw((b,r), 4, lp)
            src[b,r] = gdagpmul(Pgamma{4,1},U[bd4,4,rd4],Spinor{4,SU3fund{Float64}}(ntuple(i -> (i==s)*SU3fund{Float64}(ntuple(j -> (j==c)*1.0,3)...),4)))/2
        end

        return nothing
    end

    @timeit "Propagator computation" begin
        SF_bndfix!(pro,lp)
        fill!(dws.sp,zero(eltype(scalar_field(Spinor{4,SU3fund{Float64}},lp))))

        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_assign_bndsrc!(dws.sp, U, lp, c, s)
        end

        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnlg5!(dws.sp)
        end

        g5Dw!(pro,U,dpar.ct*dws.sp,mtwmdpar(dpar),dws,lp)

        niter = CG!(pro,U,DwdagDw!,dpar,lp,dws,maxiter,tol)
    end

    return niter
end

"""

    function Tbndpropagator!(pro, U, dpar::DiracParam{T}, dws::DiracWorkspace, lp::SpaceParm{4,6,1,D}, maxiter::Int64, tol::Float64, c::Int64, s::Int64)
        
Returns the propagator from the t=T boundary to the bulk for the SF boundary conditions for a source with color 'c' and spin 's'. The factor c_t is included while the factor 1/sqrt(V) is not.
For the propagator from t=0 to the bulk, use the function bndpropagator(U, dpar::DiracParam{T}, dws::DiracWorkspace, lp::SpaceParm{4,6,1,D}, maxiter::Int64, tol::Float64, c::Int64, s::Int64). Returns the number of iterations.

"""
function Tbndpropagator!(pro, U, dpar::DiracParam{T}, dws::DiracWorkspace, lp::SpaceParm{4,6,1,D}, maxiter::Int64, tol::Float64, c::Int64, s::Int64) where {T,D}

    function krnlg5!(src)
        b=Int64(CUDA.threadIdx().x)
        r=Int64(CUDA.blockIdx().x)
        src[b,r] = dmul(Gamma{5},src[b,r])
        return nothing
    end

    function krnl_assign_bndsrc!(src,U,lp::SpaceParm, c::Int64, s::Int64)
        b=Int64(CUDA.threadIdx().x)
        r=Int64(CUDA.blockIdx().x)

        if (point_time((b,r),lp) == lp.iL[end])
            src[b,r] = gpmul(Pgamma{4,-1},U[b,4,r],Spinor{4,SU3fund{Float64}}(ntuple(i -> (i==s)*SU3fund{Float64}(ntuple(j -> (j==c)*1.0,3)...),4)))/2
        end

        return nothing
    end

    @timeit "Propagator computation" begin
        fill!(dws.sp,zero(eltype(scalar_field(Spinor{4,SU3fund{Float64}},lp))))

        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_assign_bndsrc!(dws.sp, U, lp, c, s)
        end

        CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnlg5!(dws.sp)
        end


        g5Dw!(pro,U,dpar.ct*dws.sp,mtwmdpar(dpar),dws,lp)

        niter = CG!(pro,U,DwdagDw!,dpar,lp,dws,maxiter,tol)
    end
    return niter
end


"""
    function bndtobnd(bndpro, U, dpar, dws, lp)

Returns the boundary to boundary propagator of the Schrodinger functional given that bndpro is the propagator from t = 0 to the bulk, given by the function bndpropagator!.

"""
function bndtobnd(bndpro::AbstractArray, U::AbstractArray, dpar::DiracParam, dws::DiracWorkspace, lp::SpaceParm{4,6,1,D}) where {D}

    function krnl_bndtobnd!(psi::AbstractArray, bndp::AbstractArray, U::AbstractArray, lp::SpaceParm)
        b=Int64(CUDA.threadIdx().x)
        r=Int64(CUDA.blockIdx().x)

        if point_time((b, r), lp) == lp.iL[end]
            psi[b,r] = gdagpmul(Pgamma{4,1},U[b,4,r],bndpro[b,r])/2
        else
            psi[b,r] = 0.0*bndpro[b,r]
        end

        return nothing
    end

    CUDA.@sync begin
        CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_bndtobnd!(dws.sp, bndpro ,U, lp)
    end

    res = -dpar.ct * sum(dws.sp) / (lp.iL[1]*lp.iL[2]*lp.iL[3])

    return res
end
