
"""
    read_prop(fname::String)

Reads propagator from file `fname` using the native (BDIO) format.
"""
function read_prop(fname::String)

    UID_HDR = 14
    fb = BDIO_open(fname, "r")
    while BDIO_get_uinfo(fb) != UID_HDR
        BDIO_seek!(fb)
    end
    ihdr = Vector{Int32}(undef, 2)
    BDIO_read(fb, ihdr)
    if (ihdr[1] != convert(Int32, 1753996112)) && (ihdr[2] != convert(Int32, 5))
        error("Wrong file format [header]")
    end

    run = BDIO.BDIO_read_str(fb)

    while BDIO_get_uinfo(fb) != 1
        BDIO_seek!(fb)
    end


    ifoo = Vector{Int32}(undef, 2)
    BDIO_read(fb, ifoo)
    ndim = convert(Int64, ifoo[1])
    npls = convert(Int64, round(ndim*(ndim-1)/2))
    ibc  = convert(Int64, ifoo[2])

    ifoo = Vector{Int32}(undef, ndim+convert(Int32, npls))
    BDIO_read(fb, ifoo)
    iL  = ntuple(i -> convert(Int64, ifoo[i]),ndim)
    ntw = ntuple(i -> convert(Int64, ifoo[i+ndim]), npls)

    foopars = Vector{Float64}(undef, 4)
    BDIO_read(fb, foopars)

    footh = Vector{Float64}(undef, 4)

    lp = SpaceParm{ndim}(iL, (4,4,4,4), ibc, ntw)
    dpar = DiracParam{Float64}(SU3fund,foopars[1],foopars[2],ntuple(i -> footh[i], 4),foopars[3],foopars[4])


    dtr = (2,3,4,1)
    
    while BDIO_get_uinfo(fb) != 8
        BDIO_seek!(fb)
    end

    psicpu = Array{Spinor{4,SU3fund{Float64}}, 2}(undef, lp.bsz, lp.rsz)
    
    spvec = Vector{ComplexF64}(undef,12)

    for i4 in 1:lp.iL[4]
        for i1 in 1:lp.iL[1]
            for i2 in 1:lp.iL[2]
                for i3 in 1:lp.iL[3]
                    b, r = point_index(CartesianIndex(i1,i2,i3,i4), lp)

                    BDIO_read(fb, spvec)   
                    psicpu[b,r] = Spinor{4,SU3fund{Float64}}(ntuple(i -> SU3fund{Float64}(spvec[Int(3*i - 2)],spvec[Int(3*i - 1)] , spvec[Int(3*i)] ),4))
                
                end
            end
        end
    end

    
    BDIO_close!(fb)
    return CuArray(psicpu)    
end


"""
    save_prop(fname, psi, lp::SpaceParm, dpar::DiracParam; run::Union{Nothing,String}=nothing)

Saves propagator `psi` in the file `fname` using the native (BDIO) format.
"""
function save_prop(fname::String, psi, lp::SpaceParm{4,M,B,D}, dpar::DiracParam; run::Union{Nothing,String}=nothing) where {M,B,D}

    ihdr = [convert(Int32, 1753996112),convert(Int32,5)]
    UID_HDR = 14

    if isfile(fname)
        fb = BDIO_open(fname, "a")
    else
        fb = BDIO_open(fname, "w", "Propagator of LatticeGPU.jl")
        BDIO_start_record!(fb, BDIO_BIN_GENERIC, UID_HDR)
        BDIO_write!(fb, ihdr)
        if run != nothing
            BDIO_write!(fb, run*"\0")
        end
        BDIO_write_hash!(fb)
        
        BDIO_start_record!(fb, BDIO_BIN_GENERIC, 1)
        BDIO_write!(fb, [convert(Int32, 4)])
        BDIO_write!(fb, [convert(Int32, B)])
        BDIO_write!(fb, [convert(Int32, lp.iL[i]) for i in 1:4])
        BDIO_write!(fb, [convert(Int32, lp.ntw[i]) for i in 1:M])
        BDIO_write!(fb, [dpar.m0, dpar.csw, dpar.tm, dpar.ct])
        BDIO_write!(fb, [dpar.th[i] for i in 1:4])
    end
    BDIO_write_hash!(fb)

    dtr = (2,3,4,1)

    BDIO_start_record!(fb, BDIO_BIN_F64LE, 8, true)
    psicpu = Array(psi)

    for i4 in 1:lp.iL[4]
        for i1 in 1:lp.iL[1]
            for i2 in 1:lp.iL[2]
                for i3 in 1:lp.iL[3]
                    b, r = point_index(CartesianIndex(i1,i2,i3,i4), lp)
                    for sp in 1:4
                        for c in (:t1,:t2,:t3)
                            BDIO_write!(fb, [getfield(psicpu[b,r].s[sp],c)])
                        end
                    end
                end
            end
        end
    end

    BDIO_write_hash!(fb)
    BDIO_close!(fb)

    return nothing
end



"""
    read_dpar(fname::String)

Reads Dirac parameters from file `fname` using the native (BDIO) format. Returns DiracParam and SpaceParm.
"""
function read_dpar(fname::String)

    UID_HDR = 14
    fb = BDIO_open(fname, "r")
    while BDIO_get_uinfo(fb) != UID_HDR
        BDIO_seek!(fb)
    end
    ihdr = Vector{Int32}(undef, 2)
    BDIO_read(fb, ihdr)
    if (ihdr[1] != convert(Int32, 1753996112)) && (ihdr[2] != convert(Int32, 5))
        error("Wrong file format [header]")
    end

    run = BDIO.BDIO_read_str(fb)

    while BDIO_get_uinfo(fb) != 1
        BDIO_seek!(fb)
    end


    ifoo = Vector{Int32}(undef, 2)
    BDIO_read(fb, ifoo)
    ndim = convert(Int64, ifoo[1])
    npls = convert(Int64, round(ndim*(ndim-1)/2))
    ibc  = convert(Int64, ifoo[2])

    ifoo = Vector{Int32}(undef, ndim+convert(Int32, npls))
    BDIO_read(fb, ifoo)
    iL  = ntuple(i -> convert(Int64, ifoo[i]),ndim)
    ntw = ntuple(i -> convert(Int64, ifoo[i+ndim]), npls)

    foopars = Vector{Float64}(undef, 4)
    BDIO_read(fb, foopars)

    footh = Vector{Float64}(undef, 4)

    lp = SpaceParm{ndim}(iL, (4,4,4,4), ibc, ntw)
    dpar = DiracParam{Float64}(SU3fund,foopars[1],foopars[2],ntuple(i -> footh[i], 4),foopars[3],foopars[4])

    
    BDIO_close!(fb)
    return dpar, lp    
end
