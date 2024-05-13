using LatticeGPU, CUDA, TimerOutputs

#Test for the relation K(t,y;0,n)^+ Dw(n|m)^{-1} e^(ipm) = D(p)^{-1} exp(4t sin^2(p/2)) e^{ipn} with a given momenta (if p=0 its randomized), spin and color
#Kernel en 1207.2096


@timeit "Plw backflow test" begin

    function Dwpw_test(;p=0,s=1,c=1)
        lp = SpaceParm{4}((16,16,16,16), (4,4,4,4), 0, (0,0,0,0,0,0))
        gp = GaugeParm{Float64}(SU3{Float64}, 6.0, 1.0)
        dpar = DiracParam{Float64}(SU3fund,1.3,0.0,(1.0,1.0,1.0,1.0),0.0)
        dws = DiracWorkspace(SU3fund,Float64,lp);
        ymws = YMworkspace(SU3,Float64,lp);

        p==0 ? p = Int.(round.(lp.iL.*rand(4),RoundUp)) : nothing
        U = fill!(vector_field(SU3{Float64},lp),one(SU3{Float64}))

        rm = 2* pi* p./(lp.iL)
        rmom=(rm[1],rm[2],rm[3],rm[4])

        int = wfl_rk3(Float64, 0.01, 1.0)
        Nsteps = 15

        @timeit "Generate plane wave" begin

            pwave = fill!(scalar_field(Spinor{4,SU3fund{Float64}},lp),zero(eltype(scalar_field(Spinor{4,SU3fund{Float64}},lp))))
            prop = scalar_field(Spinor{4,SU3fund{Float64}},lp)
            prop_th = fill!(scalar_field(Spinor{4,SU3fund{Float64}},lp),zero(eltype(scalar_field(Spinor{4,SU3fund{Float64}},lp))))


            #Generate plane wave

            for x in 1:lp.iL[1] for y in 1:lp.iL[2] for z in 1:lp.iL[3] for t in 1:lp.iL[4]
                CUDA.@allowscalar pwave[point_index(CartesianIndex{lp.ndim}((x,y,z,t)),lp)...] = exp(im * (x*rmom[1] + y*rmom[2] + z*rmom[3] + t*rmom[4]))*Spinor{4,SU3fund{Float64}}(ntuple(i -> (i==s)*SU3fund{Float64}(ntuple(j -> (j==c)*1.0,3)...),4))
            end end end end

        end

        @timeit "Generate analitical solution" begin

            #Th solution

            if s == 1
                vals = (dpar.m0 + 4.0 - sum(cos.(rmom)),0.0,im*sin(rmom[4])+sin(rmom[3]),im*sin(rmom[2])+sin(rmom[1]))
                for x in 1:lp.iL[1] for y in 1:lp.iL[2] for z in 1:lp.iL[3] for t in 1:lp.iL[4]
                    CUDA.@allowscalar prop_th[point_index(CartesianIndex{lp.ndim}((x,y,z,t)),lp)...] = exp(im * (x*rmom[1] + y*rmom[2] + z*rmom[3] + t*rmom[4]))*
                        ( Spinor{4,SU3fund{Float64}}(ntuple(i -> SU3fund{Float64}(ntuple(j -> (j==c)*vals[i],3)...),4)) )/(sum((sin.(rmom)) .^2) + (dpar.m0+ 4.0 - sum(cos.(rmom)))^2)
                end end end end
            elseif s == 2
                vals = (0.0,dpar.m0 + 4.0 - sum(cos.(rmom)),sin(rmom[1]) - im *sin(rmom[2]),-sin(rmom[3])+im*sin(rmom[4]))
                for x in 1:lp.iL[1] for y in 1:lp.iL[2] for z in 1:lp.iL[3] for t in 1:lp.iL[4]
                    CUDA.@allowscalar prop_th[point_index(CartesianIndex{lp.ndim}((x,y,z,t)),lp)...] = exp(im * (x*rmom[1] + y*rmom[2] + z*rmom[3] + t*rmom[4]))*
                        ( Spinor{4,SU3fund{Float64}}(ntuple(i -> SU3fund{Float64}(ntuple(j -> (j==c)*vals[i],3)...),4)) )/(sum((sin.(rmom)) .^2) + (dpar.m0+ 4.0 - sum(cos.(rmom)))^2)
                end end end end
            elseif s == 3
                vals = (-sin(rmom[3])+im*sin(rmom[4]),-sin(rmom[1])-im*sin(rmom[2]),dpar.m0 + 4.0 - sum(cos.(rmom)),0.0)
                for x in 1:lp.iL[1] for y in 1:lp.iL[2] for z in 1:lp.iL[3] for t in 1:lp.iL[4]
                    CUDA.@allowscalar prop_th[point_index(CartesianIndex{lp.ndim}((x,y,z,t)),lp)...] = exp(im * (x*rmom[1] + y*rmom[2] + z*rmom[3] + t*rmom[4]))*
                        ( Spinor{4,SU3fund{Float64}}(ntuple(i -> SU3fund{Float64}(ntuple(j -> (j==c)*vals[i],3)...),4)) )/(sum((sin.(rmom)) .^2) + (dpar.m0+ 4.0 - sum(cos.(rmom)))^2)
                end end end end
            else
                vals = (-sin(rmom[1])+im*sin(rmom[2]),sin(rmom[3])+im*sin(rmom[4]),0.0,dpar.m0 + 4.0 - sum(cos.(rmom)))
                for x in 1:lp.iL[1] for y in 1:lp.iL[2] for z in 1:lp.iL[3] for t in 1:lp.iL[4]
                    CUDA.@allowscalar prop_th[point_index(CartesianIndex{lp.ndim}((x,y,z,t)),lp)...] = exp(im * (x*rmom[1] + y*rmom[2] + z*rmom[3] + t*rmom[4]))*
                        ( Spinor{4,SU3fund{Float64}}(ntuple(i -> SU3fund{Float64}(ntuple(j -> (j==c)*vals[i],3)...),4)) )/(sum((sin.(rmom)) .^2) + (dpar.m0+ 4.0 - sum(cos.(rmom)))^2)
                end end end end
            end

        end

        prop_th .= exp(-4*Nsteps*int.eps*sum(sin.(rmom./2).^2))*prop_th


        #compute Sum{x} D^-1(x|y)P(y)

        @timeit "Solving propagator and flowing" begin

            function krnlg5!(src)
                b=Int64(CUDA.threadIdx().x)
                r=Int64(CUDA.blockIdx().x)
                src[b,r] = dmul(Gamma{5},src[b,r])
                return nothing
            end

            CUDA.@sync begin
                CUDA.@cuda threads=lp.bsz blocks=lp.rsz krnlg5!(pwave)
            end
            g5Dw!(prop,U,pwave,dpar,dws,lp)
            CG!(prop,U,DwdagDw!,dpar,lp,dws,10000,1.0e-14)

            for _ in 1:Nsteps
                backflow(U,prop,1,int.eps,gp,dpar,lp, ymws,dws)
            end
        end


        dif = sum(norm2.(prop - prop_th))

        return dif
    end



    begin
        dif = 0.0
        for i in 1:3 for j in 1:4
            dif += Dwpw_test(c=i,s=j)
        end end

        if dif < 1.0e-5
            print("Backflow_tl test passed with average error ", dif/12,"!\n")
        else
            error("Backflow_tl test failed with difference: ",dif,"\n")
        end


    end
end
