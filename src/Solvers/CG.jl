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

"""
    function CG!

Solves the linear equation `Ax = si`
"""
function CG!(si, U, m0, A, lp::SpaceParm, dws::DiracWorkspace)

    dws.sr .= si
    dws.sp .= si
    norm = CUDA.mapreduce(x -> norm2(x), +, si)
    fill!(si,zero(eltype(so)))
    err = 0.0
    
    tol = eps * norm
    iterations = 0

    niter = 0
    for i in 1:maxiter
        A(dws.sAp, U, dws.sp, am0, dws.st, lp)
        prod  = CUDA.mapreduce(x -> dot(x[1],x[2]), +, zip(dws.sp, dws.sAp))
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
