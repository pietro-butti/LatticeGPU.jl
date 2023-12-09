###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    CG.jl
### created: Tue Nov 30 11:10:57 2021
###                               

function krnl_dot!(sum,fone,ftwo)
    b=Int64(CUDA.threadIdx().x)
    r=Int64(CUDA.blockIdx().x)

    sum[b,r] = dot(fone[b,r],ftwo[b,r])

return nothing
end

function field_dot(fone::AbstractArray,ftwo::AbstractArray,sumf,lp)
        
    CUDA.@sync begin
            CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_dot!(sumf,fone,ftwo)
    end

    return sum(sumf)
end


"""
    function CG!

Solves the linear equation `Ax = si`
"""
function CG!(si, U, A, dpar::DiracParam, lp::SpaceParm, dws::DiracWorkspace{T}, maxiter::Int64 = 10, tol=1.0) where {T}

    dws.sr .= si
    dws.sp .= si
    norm = CUDA.mapreduce(x -> norm2(x), +, si)
    fill!(si,zero(eltype(si)))
    err = 0.0

    tol = tol * norm

    iterations = 0
    sumf = scalar_field(Complex{T}, lp) 

    niter = 0
    for i in 1:maxiter
        A(dws.sAp, U, dws.sp, dpar, dws, lp)

        prod = field_dot(dws.sp,dws.sAp,sumf,lp)
        
        alpha = norm/prod

        si     .= si     .+ alpha .* dws.sp
        dws.sr .= dws.sr .- alpha .* dws.sAp

        err = CUDA.mapreduce(x -> norm2(x), +, dws.sr)
        
        if err < tol
	    niter = i
            break
        end

        beta = err/norm
        dws.sp .= dws.sr .+ beta .* dws.sp
        
        norm = err;
    end

    if err > tol
	error("CG! not converged after $maxiter iterations (Residuals: $err)")
    end
    
    return niter
end