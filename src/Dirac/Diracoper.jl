###
### "THE BEER-WARE LICENSE":
### Alberto Ramos and Fernando Panadero
### wrote this file. As long as you retain this
### notice you can do whatever you want with this stuff. If we meet some
### day, and you think this stuff is worth it, you can buy us a beer in
### return. <alberto.ramos@cern.ch> <fernando.p@csic.es>
###
### file:    Diracoper.jl
### created: Thu Nov 18 17:20:24 2021
###



## OPEN

"""
    function Dw!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::SpaceParm{4,6,B,D})

Computes the Dirac operator (with the Wilson term) `\`\``D_w``\`\` with gauge field U and parameters `dpar` of the field `si` and stores it in `so`.
If `dpar.csw` is different from zero, the clover term should be stored in `dws.csw` via the Csw! function and is automatically included in the operator.

"""
function Dw!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::SpaceParm{4,6,BC_OPEN,D}) where {D}

    SF_bndfix!(si,lp)
    if abs(dpar.csw) > 1.0E-10
        @timeit "Dw" begin
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_Dwimpr!(so, U, si, dws.csw, dpar.m0, dpar.tm, dpar.th, dpar.csw, dpar.ct, lp)
            end
        end
    else
        @timeit "Dw" begin
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_Dw!(so, U, si, dpar.m0, dpar.tm, dpar.th, dpar.ct, lp)
            end
        end
    end
    SF_bndfix!(so,lp)

    return nothing
end

function krnl_Dwimpr!(so, U, si, Fcsw, m0, tm, th, csw, ct, lp::SpaceParm{4,6,BC_OPEN,D}) where {D}

    # The field si is assumed to be zero at t = 0,T
    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    if ((point_time((b,r),lp) != 1) && (point_time((b,r),lp) != lp.iL[end]))
        bu1, ru1 = up((b,r), 1, lp)
        bd1, rd1 = dw((b,r), 1, lp)
        bu2, ru2 = up((b,r), 2, lp)
        bd2, rd2 = dw((b,r), 2, lp)
        bu3, ru3 = up((b,r), 3, lp)
        bd3, rd3 = dw((b,r), 3, lp)
        bu4, ru4 = up((b,r), 4, lp)
        bd4, rd4 = dw((b,r), 4, lp)

        @inbounds begin

            so[b,r] = (4+m0)*si[b,r]  + im*tm*dmul(Gamma{5},si[b,r]) + 0.5*csw*im*( Fcsw[b,1,r]*dmul(Gamma{10},si[b,r]) + Fcsw[b,2,r]*dmul(Gamma{11},si[b,r]) + Fcsw[b,3,r]*dmul(Gamma{12},si[b,r])
                                                                                    +Fcsw[b,4,r]*dmul(Gamma{15},si[b,r]) + Fcsw[b,5,r]*dmul(Gamma{14},si[b,r]) + Fcsw[b,6,r]*dmul(Gamma{13},si[b,r]))


            so[b,r] -= 0.5*(th[1]*gpmul(Pgamma{1,-1},U[b,1,r],si[bu1,ru1]) +conj(th[1])*gdagpmul(Pgamma{1,+1},U[bd1,1,rd1],si[bd1,rd1]) +
                th[2]*gpmul(Pgamma{2,-1},U[b,2,r],si[bu2,ru2]) +conj(th[2])*gdagpmul(Pgamma{2,+1},U[bd2,2,rd2],si[bd2,rd2]) +
                th[3]*gpmul(Pgamma{3,-1},U[b,3,r],si[bu3,ru3]) +conj(th[3])*gdagpmul(Pgamma{3,+1},U[bd3,3,rd3],si[bd3,rd3]) +
                th[4]*gpmul(Pgamma{4,-1},U[b,4,r],si[bu4,ru4]) +conj(th[4])*gdagpmul(Pgamma{4,+1},U[bd4,4,rd4],si[bd4,rd4]) )

            if (point_time((b,r),lp) == 2) || (point_time((b,r),lp) == (lp.iL[4]-1))
                so[b,r] += (ct-1.0)*si[b,r]
            end
        end
    end

    return nothing
end

function krnl_Dw!(so, U, si, m0, tm, th, ct, lp::SpaceParm{4,6,BC_OPEN,D}) where {D}

    # The field si is assumed to be zero at t = 0,T
    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    if ((point_time((b,r),lp) != 1) && (point_time((b,r),lp) != lp.iL[end]))

        bu1, ru1 = up((b,r), 1, lp)
        bd1, rd1 = dw((b,r), 1, lp)
        bu2, ru2 = up((b,r), 2, lp)
        bd2, rd2 = dw((b,r), 2, lp)
        bu3, ru3 = up((b,r), 3, lp)
        bd3, rd3 = dw((b,r), 3, lp)
        bu4, ru4 = up((b,r), 4, lp)
        bd4, rd4 = dw((b,r), 4, lp)

        @inbounds begin

            so[b,r] = (4+m0)*si[b,r] + im*tm*dmul(Gamma{5},si[b,r])
            so[b,r] -= 0.5*(th[1]*gpmul(Pgamma{1,-1},U[b,1,r],si[bu1,ru1]) +conj(th[1])*gdagpmul(Pgamma{1,+1},U[bd1,1,rd1],si[bd1,rd1]) +
                th[2]*gpmul(Pgamma{2,-1},U[b,2,r],si[bu2,ru2]) +conj(th[2])*gdagpmul(Pgamma{2,+1},U[bd2,2,rd2],si[bd2,rd2]) +
                th[3]*gpmul(Pgamma{3,-1},U[b,3,r],si[bu3,ru3]) +conj(th[3])*gdagpmul(Pgamma{3,+1},U[bd3,3,rd3],si[bd3,rd3]) +
                th[4]*gpmul(Pgamma{4,-1},U[b,4,r],si[bu4,ru4]) +conj(th[4])*gdagpmul(Pgamma{4,+1},U[bd4,4,rd4],si[bd4,rd4]) )

            if (point_time((b,r),lp) == 2) || (point_time((b,r),lp) == (lp.iL[4]-1))
                so[b,r] += (ct-1.0)*si[b,r]
            end
        end
    end

    return nothing
end


"""
    function g5Dw!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::SpaceParm{4,6,B,D})

Computes \`\` \\gamma_5 \`\` times the Dirac operator (with the Wilson term) with gauge field U and parameters `dpar` of the field `si` and stores it in `so`.
If `dpar.csw` is different from zero, the clover term should be stored in `dws.csw` via the Csw! function and is automatically included in the operator.
"""
function g5Dw!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::SpaceParm{4,6,BC_OPEN,D}) where {D}

    SF_bndfix!(si,lp)
    if abs(dpar.csw) > 1.0E-10
        @timeit "g5Dw" begin
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dwimpr!(so, U, si, dws.csw, dpar.m0, dpar.tm, dpar.th, dpar.csw, dpar.ct, lp)
            end
        end
    else
        @timeit "g5Dw" begin
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dw!(so, U, si, dpar.m0, dpar.tm, dpar.th, dpar.ct, lp)
            end
        end
    end
    SF_bndfix!(so,lp)

    return nothing
end

function krnl_g5Dwimpr!(so, U, si, Fcsw, m0, tm, th, csw, ct, lp::SpaceParm{4,6,BC_OPEN,D}) where {D}

    # The field si is assumed to be zero at t = 0,T
    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    if ((point_time((b,r),lp) != 1) && (point_time((b,r),lp) != lp.iL[end]))

        bu1, ru1 = up((b,r), 1, lp)
        bd1, rd1 = dw((b,r), 1, lp)
        bu2, ru2 = up((b,r), 2, lp)
        bd2, rd2 = dw((b,r), 2, lp)
        bu3, ru3 = up((b,r), 3, lp)
        bd3, rd3 = dw((b,r), 3, lp)
        bu4, ru4 = up((b,r), 4, lp)
        bd4, rd4 = dw((b,r), 4, lp)

        @inbounds begin

            so[b,r] = (4+m0)*si[b,r]  + 0.5*csw*im*( Fcsw[b,1,r]*dmul(Gamma{10},si[b,r]) + Fcsw[b,2,r]*dmul(Gamma{11},si[b,r]) + Fcsw[b,3,r]*dmul(Gamma{12},si[b,r])
                                                     +Fcsw[b,4,r]*dmul(Gamma{15},si[b,r]) + Fcsw[b,5,r]*dmul(Gamma{14},si[b,r]) + Fcsw[b,6,r]*dmul(Gamma{13},si[b,r]))


            so[b,r] -= 0.5*(th[1]*gpmul(Pgamma{1,-1},U[b,1,r],si[bu1,ru1]) +conj(th[1])*gdagpmul(Pgamma{1,+1},U[bd1,1,rd1],si[bd1,rd1]) +
                th[2]*gpmul(Pgamma{2,-1},U[b,2,r],si[bu2,ru2]) +conj(th[2])*gdagpmul(Pgamma{2,+1},U[bd2,2,rd2],si[bd2,rd2]) +
                th[3]*gpmul(Pgamma{3,-1},U[b,3,r],si[bu3,ru3]) +conj(th[3])*gdagpmul(Pgamma{3,+1},U[bd3,3,rd3],si[bd3,rd3]) +
                th[4]*gpmul(Pgamma{4,-1},U[b,4,r],si[bu4,ru4]) +conj(th[4])*gdagpmul(Pgamma{4,+1},U[bd4,4,rd4],si[bd4,rd4]) )

            if (point_time((b,r),lp) == 2) || (point_time((b,r),lp) == (lp.iL[4]-1))
                so[b,r] += (ct-1.0)*si[b,r]
            end
        end
    end

    so[b,r] = dmul(Gamma{5}, so[b,r])+ im*tm*si[b,r]

    return nothing
end

function krnl_g5Dw!(so, U, si, m0, tm, th, ct, lp::SpaceParm{4,6,BC_OPEN,D}) where {D}

    # The field si is assumed to be zero at t = 0,T
    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    if ((point_time((b,r),lp) != 1) && (point_time((b,r),lp) != lp.iL[end]))

        bu1, ru1 = up((b,r), 1, lp)
        bd1, rd1 = dw((b,r), 1, lp)
        bu2, ru2 = up((b,r), 2, lp)
        bd2, rd2 = dw((b,r), 2, lp)
        bu3, ru3 = up((b,r), 3, lp)
        bd3, rd3 = dw((b,r), 3, lp)
        bu4, ru4 = up((b,r), 4, lp)
        bd4, rd4 = dw((b,r), 4, lp)

        @inbounds begin

            so[b,r] = (4+m0)*si[b,r]
            so[b,r] -= 0.5*(th[1]*gpmul(Pgamma{1,-1},U[b,1,r],si[bu1,ru1]) +conj(th[1])*gdagpmul(Pgamma{1,+1},U[bd1,1,rd1],si[bd1,rd1]) +
                th[2]*gpmul(Pgamma{2,-1},U[b,2,r],si[bu2,ru2]) +conj(th[2])*gdagpmul(Pgamma{2,+1},U[bd2,2,rd2],si[bd2,rd2]) +
                th[3]*gpmul(Pgamma{3,-1},U[b,3,r],si[bu3,ru3]) +conj(th[3])*gdagpmul(Pgamma{3,+1},U[bd3,3,rd3],si[bd3,rd3]) +
                th[4]*gpmul(Pgamma{4,-1},U[b,4,r],si[bu4,ru4]) +conj(th[4])*gdagpmul(Pgamma{4,+1},U[bd4,4,rd4],si[bd4,rd4]) )

            if (point_time((b,r),lp) == 2) || (point_time((b,r),lp) == (lp.iL[4]-1))
                so[b,r] += (ct-1.0)*si[b,r]
            end
        end
    end

    so[b,r] = dmul(Gamma{5}, so[b,r]) + im*tm*si[b,r]

    return nothing
end

"""
    function DwdagDw!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::SpaceParm{4,6,B,D})

Applies the operator \`\` \\gamma_5 D_w \`\` twice to `si` and stores the result in `so`. This is equivalent to appling the operator \`\` D_w^\\dagger D_w \`\`
The Dirac operator is the same as in the functions `Dw!` and `g5Dw!`
"""
function DwdagDw!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::SpaceParm{4,6,BC_OPEN,D}) where {D}

    SF_bndfix!(si,lp)
    if abs(dpar.csw) > 1.0E-10
        @timeit "DwdagDw" begin

            @timeit "g5Dw" begin
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dwimpr!(dws.st, U, si, dws.csw, dpar.m0, dpar.tm, dpar.th, dpar.csw, dpar.ct, lp)
                end
            end
            SF_bndfix!(dws.st,lp)
            @timeit "g5Dw" begin
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dwimpr!(so, U, dws.st, dws.csw, dpar.m0, -dpar.tm, dpar.th, dpar.csw, dpar.ct, lp)
                end
            end
            SF_bndfix!(so,lp)
        end
    else
        @timeit "DwdagDw" begin

            @timeit "g5Dw" begin
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dw!(dws.st, U, si, dpar.m0, dpar.tm, dpar.th, dpar.ct, lp)
                end
            end
            SF_bndfix!(dws.st,lp)
            @timeit "g5Dw" begin
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dw!(so, U, dws.st, dpar.m0, -dpar.tm, dpar.th, dpar.ct, lp)
                end
            end
            SF_bndfix!(so,lp)
        end
    end

    return nothing
end

## PERDIODIC

function Dw!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::SpaceParm{4,6,BC_PERIODIC,D}) where {D}

    if abs(dpar.csw) > 1.0E-10
        @timeit "Dw" begin
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_Dwimpr!(so, U, si, dws.csw, dpar.m0, dpar.tm, dpar.th, dpar.csw, lp)
            end
        end
    else
        @timeit "Dw" begin
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_Dw!(so, U, si, dpar.m0, dpar.tm, dpar.th, lp)
            end
        end
    end

    return nothing
end

function krnl_Dwimpr!(so, U, si, Fcsw, m0, tm, th, csw, lp::SpaceParm{4,6,BC_PERIODIC,D}) where {D}

    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    bu1, ru1 = up((b,r), 1, lp)
    bd1, rd1 = dw((b,r), 1, lp)
    bu2, ru2 = up((b,r), 2, lp)
    bd2, rd2 = dw((b,r), 2, lp)
    bu3, ru3 = up((b,r), 3, lp)
    bd3, rd3 = dw((b,r), 3, lp)
    bu4, ru4 = up((b,r), 4, lp)
    bd4, rd4 = dw((b,r), 4, lp)

    @inbounds begin

        so[b,r] = (4+m0)*si[b,r]+ im*tm*dmul(Gamma{5},si[b,r]) + 0.5*csw*im*( Fcsw[b,1,r]*dmul(Gamma{10},si[b,r]) + Fcsw[b,2,r]*dmul(Gamma{11},si[b,r]) + Fcsw[b,3,r]*dmul(Gamma{12},si[b,r])
                                                                              +Fcsw[b,4,r]*dmul(Gamma{15},si[b,r]) + Fcsw[b,5,r]*dmul(Gamma{14},si[b,r]) + Fcsw[b,6,r]*dmul(Gamma{13},si[b,r]))

        so[b,r] -= 0.5*(th[1]*gpmul(Pgamma{1,-1},U[b,1,r],si[bu1,ru1]) +conj(th[1])*gdagpmul(Pgamma{1,+1},U[bd1,1,rd1],si[bd1,rd1]) +
            th[2]*gpmul(Pgamma{2,-1},U[b,2,r],si[bu2,ru2]) +conj(th[2])*gdagpmul(Pgamma{2,+1},U[bd2,2,rd2],si[bd2,rd2]) +
            th[3]*gpmul(Pgamma{3,-1},U[b,3,r],si[bu3,ru3]) +conj(th[3])*gdagpmul(Pgamma{3,+1},U[bd3,3,rd3],si[bd3,rd3]) +
            th[4]*gpmul(Pgamma{4,-1},U[b,4,r],si[bu4,ru4]) +conj(th[4])*gdagpmul(Pgamma{4,+1},U[bd4,4,rd4],si[bd4,rd4]) )

    end

    return nothing
end

function krnl_Dw!(so, U, si, m0, tm, th, lp::SpaceParm{4,6,BC_PERIODIC,D}) where {D}

    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    bu1, ru1 = up((b,r), 1, lp)
    bd1, rd1 = dw((b,r), 1, lp)
    bu2, ru2 = up((b,r), 2, lp)
    bd2, rd2 = dw((b,r), 2, lp)
    bu3, ru3 = up((b,r), 3, lp)
    bd3, rd3 = dw((b,r), 3, lp)
    bu4, ru4 = up((b,r), 4, lp)
    bd4, rd4 = dw((b,r), 4, lp)

    @inbounds begin

        so[b,r] = (4+m0)*si[b,r] + im*tm*dmul(Gamma{5},si[b,r])

        so[b,r] -= 0.5*(th[1]*gpmul(Pgamma{1,-1},U[b,1,r],si[bu1,ru1]) +conj(th[1])*gdagpmul(Pgamma{1,+1},U[bd1,1,rd1],si[bd1,rd1]) +
            th[2]*gpmul(Pgamma{2,-1},U[b,2,r],si[bu2,ru2]) +conj(th[2])*gdagpmul(Pgamma{2,+1},U[bd2,2,rd2],si[bd2,rd2]) +
            th[3]*gpmul(Pgamma{3,-1},U[b,3,r],si[bu3,ru3]) +conj(th[3])*gdagpmul(Pgamma{3,+1},U[bd3,3,rd3],si[bd3,rd3]) +
            th[4]*gpmul(Pgamma{4,-1},U[b,4,r],si[bu4,ru4]) +conj(th[4])*gdagpmul(Pgamma{4,+1},U[bd4,4,rd4],si[bd4,rd4]) )

    end

    return nothing
end

function g5Dw!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::SpaceParm{4,6,BC_PERIODIC,D}) where {D}

    if abs(dpar.csw) > 1.0E-10
        @timeit "g5Dw" begin
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dwimpr!(so, U, si, dws.csw, dpar.m0, dpar.tm, dpar.th, dpar.csw, lp)
            end
        end
    else
        @timeit "g5Dw" begin
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dw!(so, U, si, dpar.m0, dpar.tm, dpar.th, lp)
            end
        end
    end

    return nothing
end

function krnl_g5Dwimpr!(so, U, si, Fcsw, m0, tm, th, csw, lp::SpaceParm{4,6,BC_PERIODIC,D}) where {D}

    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    bu1, ru1 = up((b,r), 1, lp)
    bd1, rd1 = dw((b,r), 1, lp)
    bu2, ru2 = up((b,r), 2, lp)
    bd2, rd2 = dw((b,r), 2, lp)
    bu3, ru3 = up((b,r), 3, lp)
    bd3, rd3 = dw((b,r), 3, lp)
    bu4, ru4 = up((b,r), 4, lp)
    bd4, rd4 = dw((b,r), 4, lp)

    @inbounds begin

        so[b,r] = (4+m0)*si[b,r]  + 0.5*csw*im*( Fcsw[b,1,r]*dmul(Gamma{10},si[b,r]) + Fcsw[b,2,r]*dmul(Gamma{11},si[b,r]) + Fcsw[b,3,r]*dmul(Gamma{12},si[b,r])
                                                 +Fcsw[b,4,r]*dmul(Gamma{15},si[b,r]) + Fcsw[b,5,r]*dmul(Gamma{14},si[b,r]) + Fcsw[b,6,r]*dmul(Gamma{13},si[b,r]))

        so[b,r] -= 0.5*(th[1]*gpmul(Pgamma{1,-1},U[b,1,r],si[bu1,ru1]) +conj(th[1])*gdagpmul(Pgamma{1,+1},U[bd1,1,rd1],si[bd1,rd1]) +
            th[2]*gpmul(Pgamma{2,-1},U[b,2,r],si[bu2,ru2]) +conj(th[2])*gdagpmul(Pgamma{2,+1},U[bd2,2,rd2],si[bd2,rd2]) +
            th[3]*gpmul(Pgamma{3,-1},U[b,3,r],si[bu3,ru3]) +conj(th[3])*gdagpmul(Pgamma{3,+1},U[bd3,3,rd3],si[bd3,rd3]) +
            th[4]*gpmul(Pgamma{4,-1},U[b,4,r],si[bu4,ru4]) +conj(th[4])*gdagpmul(Pgamma{4,+1},U[bd4,4,rd4],si[bd4,rd4]) )

        so[b,r] = dmul(Gamma{5}, so[b,r])+ im*tm*si[b,r]
    end

    return nothing
end

function krnl_g5Dw!(so, U, si, m0, tm, th, lp::SpaceParm{4,6,BC_PERIODIC,D}) where {D}

    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    bu1, ru1 = up((b,r), 1, lp)
    bd1, rd1 = dw((b,r), 1, lp)
    bu2, ru2 = up((b,r), 2, lp)
    bd2, rd2 = dw((b,r), 2, lp)
    bu3, ru3 = up((b,r), 3, lp)
    bd3, rd3 = dw((b,r), 3, lp)
    bu4, ru4 = up((b,r), 4, lp)
    bd4, rd4 = dw((b,r), 4, lp)

    @inbounds begin

        so[b,r] = (4+m0)*si[b,r]

        so[b,r] -= 0.5*(th[1]*gpmul(Pgamma{1,-1},U[b,1,r],si[bu1,ru1]) +conj(th[1])*gdagpmul(Pgamma{1,+1},U[bd1,1,rd1],si[bd1,rd1]) +
            th[2]*gpmul(Pgamma{2,-1},U[b,2,r],si[bu2,ru2]) +conj(th[2])*gdagpmul(Pgamma{2,+1},U[bd2,2,rd2],si[bd2,rd2]) +
            th[3]*gpmul(Pgamma{3,-1},U[b,3,r],si[bu3,ru3]) +conj(th[3])*gdagpmul(Pgamma{3,+1},U[bd3,3,rd3],si[bd3,rd3]) +
            th[4]*gpmul(Pgamma{4,-1},U[b,4,r],si[bu4,ru4]) +conj(th[4])*gdagpmul(Pgamma{4,+1},U[bd4,4,rd4],si[bd4,rd4]) )

        so[b,r] = dmul(Gamma{5}, so[b,r]) + im*tm*si[b,r]
    end

    return nothing
end

function DwdagDw!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::SpaceParm{4,6,BC_PERIODIC,D}) where {D}

    if abs(dpar.csw) > 1.0E-10
        @timeit "DwdagDw" begin

            @timeit "g5Dw" begin
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dwimpr!(dws.st, U, si, dws.csw, dpar.m0, dpar.tm, dpar.th, dpar.csw, lp)
                end
            end

            @timeit "g5Dw" begin
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dwimpr!(so, U, dws.st, dws.csw, dpar.m0, -dpar.tm, dpar.th, dpar.csw, lp)
                end
            end
        end
    else
        @timeit "DwdagDw" begin

            @timeit "g5Dw" begin
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dw!(dws.st, U, si, dpar.m0, dpar.tm, dpar.th, lp)
                end
            end

            @timeit "g5Dw" begin
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dw!(so, U, dws.st, dpar.m0, -dpar.tm, dpar.th, lp)
                end
            end
        end end

    return nothing
end

## SF

function Dw!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::Union{SpaceParm{4,6,BC_SF_ORBI,D},SpaceParm{4,6,BC_SF_AFWB,D}}) where {D}

    SF_bndfix!(si,lp)
    if abs(dpar.csw) > 1.0E-10
        @timeit "Dw" begin
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_Dwimpr!(so, U, si, dws.csw, dpar.m0, dpar.tm, dpar.th, dpar.csw, dpar.ct, lp)
            end
        end
    else
        @timeit "Dw" begin
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_Dw!(so, U, si, dpar.m0, dpar.tm, dpar.th, dpar.ct, lp)
            end
        end
    end
    SF_bndfix!(so,lp)

    return nothing
end

function krnl_Dwimpr!(so, U, si, Fcsw, m0, tm, th, csw, ct, lp::Union{SpaceParm{4,6,BC_SF_ORBI,D},SpaceParm{4,6,BC_SF_AFWB,D}}) where {D}

    # The field si is assumed to be zero at t = 0

    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    if (point_time((b,r),lp) != 1)

        bu1, ru1 = up((b,r), 1, lp)
        bd1, rd1 = dw((b,r), 1, lp)
        bu2, ru2 = up((b,r), 2, lp)
        bd2, rd2 = dw((b,r), 2, lp)
        bu3, ru3 = up((b,r), 3, lp)
        bd3, rd3 = dw((b,r), 3, lp)
        bu4, ru4 = up((b,r), 4, lp)
        bd4, rd4 = dw((b,r), 4, lp)

        @inbounds begin

            so[b,r] = (4+m0)*si[b,r]  + im*tm*dmul(Gamma{5},si[b,r]) + 0.5*csw*im*( Fcsw[b,1,r]*dmul(Gamma{10},si[b,r]) + Fcsw[b,2,r]*dmul(Gamma{11},si[b,r]) + Fcsw[b,3,r]*dmul(Gamma{12},si[b,r])
                                                                                    +Fcsw[b,4,r]*dmul(Gamma{15},si[b,r]) + Fcsw[b,5,r]*dmul(Gamma{14},si[b,r]) + Fcsw[b,6,r]*dmul(Gamma{13},si[b,r]))


            so[b,r] -= 0.5*(th[1]*gpmul(Pgamma{1,-1},U[b,1,r],si[bu1,ru1]) +conj(th[1])*gdagpmul(Pgamma{1,+1},U[bd1,1,rd1],si[bd1,rd1]) +
                th[2]*gpmul(Pgamma{2,-1},U[b,2,r],si[bu2,ru2]) +conj(th[2])*gdagpmul(Pgamma{2,+1},U[bd2,2,rd2],si[bd2,rd2]) +
                th[3]*gpmul(Pgamma{3,-1},U[b,3,r],si[bu3,ru3]) +conj(th[3])*gdagpmul(Pgamma{3,+1},U[bd3,3,rd3],si[bd3,rd3]) +
                th[4]*gpmul(Pgamma{4,-1},U[b,4,r],si[bu4,ru4]) +conj(th[4])*gdagpmul(Pgamma{4,+1},U[bd4,4,rd4],si[bd4,rd4]) )

            if (point_time((b,r),lp) == 2) || (point_time((b,r),lp) == lp.iL[4])
                so[b,r] += (ct-1.0)*si[b,r]
            end
        end
    end

    return nothing
end

function krnl_Dw!(so, U, si, m0, tm, th, ct, lp::Union{SpaceParm{4,6,BC_SF_ORBI,D},SpaceParm{4,6,BC_SF_AFWB,D}}) where {D}

    # The field si is assumed to be zero at t = 0

    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    if (point_time((b,r),lp) != 1)

        bu1, ru1 = up((b,r), 1, lp)
        bd1, rd1 = dw((b,r), 1, lp)
        bu2, ru2 = up((b,r), 2, lp)
        bd2, rd2 = dw((b,r), 2, lp)
        bu3, ru3 = up((b,r), 3, lp)
        bd3, rd3 = dw((b,r), 3, lp)
        bu4, ru4 = up((b,r), 4, lp)
        bd4, rd4 = dw((b,r), 4, lp)

        @inbounds begin

            so[b,r] = (4+m0)*si[b,r] + im*tm*dmul(Gamma{5},si[b,r])
            so[b,r] -= 0.5*(th[1]*gpmul(Pgamma{1,-1},U[b,1,r],si[bu1,ru1]) +conj(th[1])*gdagpmul(Pgamma{1,+1},U[bd1,1,rd1],si[bd1,rd1]) +
                th[2]*gpmul(Pgamma{2,-1},U[b,2,r],si[bu2,ru2]) +conj(th[2])*gdagpmul(Pgamma{2,+1},U[bd2,2,rd2],si[bd2,rd2]) +
                th[3]*gpmul(Pgamma{3,-1},U[b,3,r],si[bu3,ru3]) +conj(th[3])*gdagpmul(Pgamma{3,+1},U[bd3,3,rd3],si[bd3,rd3]) +
                th[4]*gpmul(Pgamma{4,-1},U[b,4,r],si[bu4,ru4]) +conj(th[4])*gdagpmul(Pgamma{4,+1},U[bd4,4,rd4],si[bd4,rd4]) )

            if (point_time((b,r),lp) == 2) || (point_time((b,r),lp) == lp.iL[4])
                so[b,r] += (ct-1.0)*si[b,r]
            end
        end
    end

    return nothing
end


function g5Dw!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::Union{SpaceParm{4,6,BC_SF_ORBI,D},SpaceParm{4,6,BC_SF_AFWB,D}}) where {D}

    SF_bndfix!(si,lp)
    if abs(dpar.csw) > 1.0E-10
        @timeit "g5Dw" begin
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dwimpr!(so, U, si, dws.csw, dpar.m0, dpar.tm, dpar.th, dpar.csw, dpar.ct, lp)
            end
        end
    else
        @timeit "g5Dw" begin
            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dw!(so, U, si, dpar.m0, dpar.tm, dpar.th, dpar.ct, lp)
            end
        end
    end
    SF_bndfix!(so,lp)

    return nothing
end

function krnl_g5Dwimpr!(so, U, si, Fcsw, m0, tm, th, csw, ct, lp::Union{SpaceParm{4,6,BC_SF_ORBI,D},SpaceParm{4,6,BC_SF_AFWB,D}}) where {D}

    # The field si is assumed to be zero at t = 0

    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    if (point_time((b,r),lp) != 1)

        bu1, ru1 = up((b,r), 1, lp)
        bd1, rd1 = dw((b,r), 1, lp)
        bu2, ru2 = up((b,r), 2, lp)
        bd2, rd2 = dw((b,r), 2, lp)
        bu3, ru3 = up((b,r), 3, lp)
        bd3, rd3 = dw((b,r), 3, lp)
        bu4, ru4 = up((b,r), 4, lp)
        bd4, rd4 = dw((b,r), 4, lp)

        @inbounds begin

            so[b,r] = (4+m0)*si[b,r]  + 0.5*csw*im*( Fcsw[b,1,r]*dmul(Gamma{10},si[b,r]) + Fcsw[b,2,r]*dmul(Gamma{11},si[b,r]) + Fcsw[b,3,r]*dmul(Gamma{12},si[b,r])
                                                     +Fcsw[b,4,r]*dmul(Gamma{15},si[b,r]) + Fcsw[b,5,r]*dmul(Gamma{14},si[b,r]) + Fcsw[b,6,r]*dmul(Gamma{13},si[b,r]))


            so[b,r] -= 0.5*(th[1]*gpmul(Pgamma{1,-1},U[b,1,r],si[bu1,ru1]) +conj(th[1])*gdagpmul(Pgamma{1,+1},U[bd1,1,rd1],si[bd1,rd1]) +
                th[2]*gpmul(Pgamma{2,-1},U[b,2,r],si[bu2,ru2]) +conj(th[2])*gdagpmul(Pgamma{2,+1},U[bd2,2,rd2],si[bd2,rd2]) +
                th[3]*gpmul(Pgamma{3,-1},U[b,3,r],si[bu3,ru3]) +conj(th[3])*gdagpmul(Pgamma{3,+1},U[bd3,3,rd3],si[bd3,rd3]) +
                th[4]*gpmul(Pgamma{4,-1},U[b,4,r],si[bu4,ru4]) +conj(th[4])*gdagpmul(Pgamma{4,+1},U[bd4,4,rd4],si[bd4,rd4]) )

            if (point_time((b,r),lp) == 2) || (point_time((b,r),lp) == lp.iL[4])
                so[b,r] += (ct-1.0)*si[b,r]
            end
        end
    end

    so[b,r] = dmul(Gamma{5}, so[b,r])+ im*tm*si[b,r]

    return nothing
end

function krnl_g5Dw!(so, U, si, m0, tm, th, ct, lp::Union{SpaceParm{4,6,BC_SF_ORBI,D},SpaceParm{4,6,BC_SF_AFWB,D}}) where {D}

    # The field si is assumed to be zero at t = 0

    b = Int64(CUDA.threadIdx().x);  r = Int64(CUDA.blockIdx().x)

    if (point_time((b,r),lp) != 1)

        bu1, ru1 = up((b,r), 1, lp)
        bd1, rd1 = dw((b,r), 1, lp)
        bu2, ru2 = up((b,r), 2, lp)
        bd2, rd2 = dw((b,r), 2, lp)
        bu3, ru3 = up((b,r), 3, lp)
        bd3, rd3 = dw((b,r), 3, lp)
        bu4, ru4 = up((b,r), 4, lp)
        bd4, rd4 = dw((b,r), 4, lp)

        @inbounds begin

            so[b,r] = (4+m0)*si[b,r]
            so[b,r] -= 0.5*(th[1]*gpmul(Pgamma{1,-1},U[b,1,r],si[bu1,ru1]) +conj(th[1])*gdagpmul(Pgamma{1,+1},U[bd1,1,rd1],si[bd1,rd1]) +
                th[2]*gpmul(Pgamma{2,-1},U[b,2,r],si[bu2,ru2]) +conj(th[2])*gdagpmul(Pgamma{2,+1},U[bd2,2,rd2],si[bd2,rd2]) +
                th[3]*gpmul(Pgamma{3,-1},U[b,3,r],si[bu3,ru3]) +conj(th[3])*gdagpmul(Pgamma{3,+1},U[bd3,3,rd3],si[bd3,rd3]) +
                th[4]*gpmul(Pgamma{4,-1},U[b,4,r],si[bu4,ru4]) +conj(th[4])*gdagpmul(Pgamma{4,+1},U[bd4,4,rd4],si[bd4,rd4]) )

            if (point_time((b,r),lp) == 2) || (point_time((b,r),lp) == lp.iL[4])
                so[b,r] += (ct-1.0)*si[b,r]
            end
        end
    end

    so[b,r] = dmul(Gamma{5}, so[b,r]) + im*tm*si[b,r]

    return nothing
end

function DwdagDw!(so, U, si, dpar::DiracParam, dws::DiracWorkspace, lp::Union{SpaceParm{4,6,BC_SF_ORBI,D},SpaceParm{4,6,BC_SF_AFWB,D}}) where {D}

    SF_bndfix!(si,lp)
    if abs(dpar.csw) > 1.0E-10
        @timeit "DwdagDw" begin

            @timeit "g5Dw" begin
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dwimpr!(dws.st, U, si, dws.csw, dpar.m0, dpar.tm, dpar.th, dpar.csw, dpar.ct, lp)
                end
            end
            SF_bndfix!(dws.st,lp)
            @timeit "g5Dw" begin
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dwimpr!(so, U, dws.st, dws.csw, dpar.m0, -dpar.tm, dpar.th, dpar.csw, dpar.ct, lp)
                end
            end
            SF_bndfix!(so,lp)
        end
    else
        @timeit "DwdagDw" begin

            @timeit "g5Dw" begin
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dw!(dws.st, U, si, dpar.m0, dpar.tm, dpar.th, dpar.ct, lp)
                end
            end
            SF_bndfix!(dws.st,lp)
            @timeit "g5Dw" begin
                CUDA.@sync begin
                    CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnl_g5Dw!(so, U, dws.st, dpar.m0, -dpar.tm, dpar.th, dpar.ct, lp)
                end
            end
            SF_bndfix!(so,lp)
        end
    end

    return nothing
end
