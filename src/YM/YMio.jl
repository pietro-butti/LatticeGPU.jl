###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    YMio.jl
### created: Wed Nov 10 12:58:27 2021
###                               

"""
    function import_lex64(fname::String, lp::SpaceParm)

import a double precision configuration in lexicographic format. SF boundary conditions are assummed. 
"""
function import_lex64(fname, lp::SpaceParm)

    fp = open(fname, "r")

    dtr = [2,3,4,1]

    assign(id, V, i3) = SU3{Float64}(V[1,dtr[id],i3],V[2,dtr[id],i3],V[3,dtr[id],i3],
                                     V[4,dtr[id],i3],V[5,dtr[id],i3],V[6,dtr[id],i3])
    
    Ucpu = Array{SU3{Float64}, 3}(undef, lp.bsz, lp.ndim, lp.rsz)
    V = Array{ComplexF64, 3}(undef, 9, lp.ndim, lp.iL[3])
    for i4 in 1:lp.iL[4]
        for i1 in 1:lp.iL[1]
            for i2 in 1:lp.iL[2]
                read!(fp, V)
                for i3 in 1:lp.iL[3]
                    b, r = point_index(CartesianIndex(i1,i2,i3,i4), lp)
                    for id in 1:lp.ndim
                        Ucpu[b,id,r] = assign(id, V, i3)
                    end
                end
            end
        end
    end

    read!(fp, V)
    Ubnd = ntuple(i->assign(i, V, 1), 3)
    close(fp)

    return CuArray(Ucpu), Ubnd
end

"""
    function import_cern64(fname::String, ibc, lp::SpaceParm)

import a double precision configuration in CERN format. 
"""
function import_cern64(fname, ibc, lp::SpaceParm; log=true)

    fp = open(fname, "r")
    iL = Vector{Int32}(undef, 4)
    read!(fp, iL)
    avgpl = Vector{Float64}(undef, 1)
    read!(fp, avgpl)
    if log
        println("# [import_cern64] Read from conf file: ", iL, " (plaq: ", avgpl, ")")
    end
    
    dtr = [4,1,2,3]
    assign(V, ic) = SU3{Float64}(V[1,ic],V[2,ic],V[3,ic],V[4,ic],V[5,ic],V[6,ic])
    
    Ucpu = Array{SU3{Float64}, 3}(undef, lp.bsz, lp.ndim, lp.rsz)
    V = Array{ComplexF64, 2}(undef, 9, 2)
    for i4 in 1:lp.iL[4]
        for i1 in 1:lp.iL[1]
            for i2 in 1:lp.iL[2]
                for i3 in 1:lp.iL[3]
                    if (mod(i1+i2+i3+i4-4, 2) == 1)
                        b, r = point_index(CartesianIndex(i1,i2,i3,i4), lp)
                        for id in 1:lp.ndim
                            read!(fp, V)
                            Ucpu[b,dtr[id],r] = assign(V, 1)

                            bd, rd = dw((b,r), dtr[id], lp)
                            Ucpu[bd,dtr[id],rd] = assign(V, 2)
                        end
                    end
                end
            end
        end
    end
    close(fp)

    return CuArray(Ucpu)
end
