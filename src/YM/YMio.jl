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

    dtr = [4,1,2,3]
    
    Ucpu = Array{SU3{Float64}, 3}(undef, lp.bsz, lp.ndim, lp.rsz)
    Ubnd = Array{SU3{Float64}, 1}(undef, lp.ndim-1)
    V = Array{ComplexF64, 3}(undef, 9, lp.ndim, lp.iL[3])
    for i4 in 1:lp.iL[4]
        for i1 in 1:lp.iL[1]
            for i2 in 1:lp.iL[2]
                read!(fp, V)
                for i3 in 1:lp.iL[3]
                    b, r = point_index(CartesianIndex(i1,i2,i3,i4), lp)
                    for id in 1:lp.ndim
                        Ucpu[b,dtr[id],r] = SU3{Float64}(V[1,id,i4],V[2,id,i4],V[3,id,i4],
                                                         V[4,id,i4],V[5,id,i4],V[6,id,i4])
                    end
                end
            end
        end
    end

    read!(fp, V)
    for id in 2:lp.ndim
        Ubnd[dtr[id]] = SU3{Float64}(V[1,id,1],V[2,id,1],V[3,id,1],
                                     V[4,id,1],V[5,id,1],V[6,id,1])
    end

    close(fp)

    return CuArray(Ucpu), Ubnd
end
