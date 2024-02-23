using LatticeGPU
using CUDA
using TimerOutputs

@timeit "fA_fP test" begin


    function fP_test(;theta = (0.5,0.7,1.0,0.0), m = 1.3, size = (8,8,8,16),prec = 1.0e-16)

        @timeit "fP inversion (x12)" begin

            lp = SpaceParm{4}(size,(4,4,4,4),1,(0,0,0,0,0,0));
            exptheta = exp.(im.*theta./lp.iL);
            dpar = DiracParam{Float64}(SU3fund,m,0.0,exptheta,0.0,1.0);
            dws = DiracWorkspace(SU3fund,Float64,lp);

            U = fill!(vector_field(SU3{Float64},lp),one(SU3{Float64}));
            psi = scalar_field(Spinor{4,SU3fund{Float64}},lp);

            res = zeros(lp.iL[4])

            for s in 1:4 for c in 1:3
                bndpropagator!(psi,U,dpar,dws,lp,1000,prec,c,s);

                for t in 1:lp.iL[4]
                    #for i in 1:lp.iL[1]    for j in 1:lp.iL[2]        for k in 1:lp.iL[3]
                    i=abs(rand(Int))%lp.iL[1] +1;j=abs(rand(Int))%lp.iL[2] +1;k=abs(rand(Int))%lp.iL[3] +1;
                    CUDA.@allowscalar (res[t] += norm2(psi[point_index(CartesianIndex{lp.ndim}((i,j,k,t)),lp)...])/2)
                    #end end end
                    #res[t] = res[t]/(lp.iL[1]*lp.iL[2]*lp.iL[3])

                end

            end end

        end

        @timeit "fP analitical solution" begin

        #THEORETICAL SOLUTION: hep-lat/9606016 eq (2.33)

            res_th = zeros(lp.iL[4])

            pp3 = ntuple(i -> theta[i]/lp.iL[i],3)
            omega = 2 * asinh(0.5* sqrt(( sum((sin.(pp3)).^2) + (m + 2*(sum((sin.(pp3./2)).^2) ))^2) / (1+m+2*(sum((sin.(pp3./2)).^2) )) ) )
            pp = (-im*omega,pp3...)
            Mpp = m + 2* sum((sin.(pp./2)).^2)
            Rpp = Mpp*(1-exp(-2*omega*lp.iL[4])) + sinh(omega) * (1+exp(-2*omega*lp.iL[4]))

            for i in 2:lp.iL[4]
                res_th[i] = (2*3*sinh(omega)/(Rpp^2)) * ( (Mpp + sinh(omega))*exp(-2*omega*(i-1)) - (Mpp - sinh(omega))*exp(-2*omega*(2*lp.iL[4]- (i - 1))) )
            end

        end
        return sum(abs.(res-res_th))

    end

    function fA_test(;theta = (0.5,0.7,1.0,0.0), m = 1.3, size = (8,8,8,16),prec = 1.0e-16)

        @timeit "fA inversion (x12)" begin

            lp = SpaceParm{4}(size,(4,4,4,4),1,(0,0,0,0,0,0));
            exptheta = exp.(im.*theta./lp.iL);

            dpar = DiracParam{Float64}(SU3fund,m,0.0,exptheta,0.0,1.0);
            dws = DiracWorkspace(SU3fund,Float64,lp);

            U = fill!(vector_field(SU3{Float64},lp),one(SU3{Float64}));
            psi = scalar_field(Spinor{4,SU3fund{Float64}},lp);

            res = im*zeros(lp.iL[4])

            for s in 1:4 for c in 1:3
                bndpropagator!(psi,U,dpar,dws,lp,1000,prec,c,s);

                for t in 1:lp.iL[4]
                    #for i in 1:lp.iL[1]    for j in 1:lp.iL[2]        for k in 1:lp.iL[3]
                    i=abs(rand(Int))%lp.iL[1] +1;j=abs(rand(Int))%lp.iL[2] +1;k=abs(rand(Int))%lp.iL[3] +1;
                    CUDA.@allowscalar (res[t] += -dot(psi[point_index(CartesianIndex{lp.ndim}((i,j,k,t)),lp)...],dmul(Gamma{4},psi[point_index(CartesianIndex{lp.ndim}((i,j,k,t)),lp)...]))/2)
                    #end end end
                    #res[t] = res[t]/(lp.iL[1]*lp.iL[2]*lp.iL[3])

                end

            end end

        end
        #THEORETICAL SOLUTION: hep-lat/9606016 eq (2.32)

        @timeit "fA analitical solution" begin
            res_th = zeros(lp.iL[4])

            pp3 = ntuple(i -> theta[i]/lp.iL[i],3)
            omega = 2 * asinh(0.5* sqrt(( sum((sin.(pp3)).^2) + (m + 2*(sum((sin.(pp3./2)).^2) ))^2) / (1+m+2*(sum((sin.(pp3./2)).^2) )) ) )
            pp = (-im*omega,pp3...)
            Mpp = m + 2* sum((sin.(pp./2)).^2)
            Rpp = Mpp*(1-exp(-2*omega*lp.iL[4])) + sinh(omega) * (1+exp(-2*omega*lp.iL[4]))

            for i in 2:lp.iL[4]
                res_th[i] = (6/(Rpp^2)) * ( 2*(Mpp - sinh(omega))*(Mpp + sinh(omega))*exp(-2*omega*lp.iL[4])
                                            - Mpp*((Mpp + sinh(omega))*exp(-2*omega*(i-1)) + (Mpp - sinh(omega))*exp(-2*omega*(2*lp.iL[4]- (i - 1)))))
            end

        end
        return sum(abs.(res-res_th))
    end


    difA = fA_test();
    difP = fP_test();

        if difA > 1.0e-15
            error("fA test failed with error ", difA)
        elseif difP > 1.0e-15
            error("fP test failed with error ", difP)
        else
            print("fA & fP tests passed with errors: ", difA," and ",difP,"!\n")
        end

end
