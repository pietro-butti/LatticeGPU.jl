module BackendInterface
    using CUDA, AMDGPU

    abstract type AbstractBackend end
    struct CUDABackend <: AbstractBackend end
    struct AMDGPUBackend <: AbstractBackend end
    struct CPUBackend <: AbstractBackend end

    const BACKEND = Ref{AbstractBackend}(CUDABackend())

    function set_backend!(b::AbstractBackend)
        BACKEND[] = b
    end


    # ---------------- Generic algebra dispatch ------------------
    backend_sin(x) = BACKEND[] isa CUDABackend   ? CUDA.sin(x) :
                BACKEND[]  isa AMDGPUBackend ? AMDGPU.sin(x) :
                BACKEND[]  isa CPUBackend    ? sin(x) :
                error("Unsupported backend")

    backend_cos(x) = BACKEND[] isa CUDABackend   ? CUDA.cos(x) :
                BACKEND[]  isa AMDGPUBackend ? AMDGPU.cos(x) :
                BACKEND[]  isa CPUBackend    ? cos(x) :
                error("Unsupported backend")



    # --------------- Array allocation dispatch -------------------
    allocate_array(::CPUBackend   , ::Type{T}, dims::NTuple{N,Int}) where {T,N} = Array{T,N}(undef, dims...)
    allocate_array(::CUDABackend  , ::Type{T}, dims::NTuple{N,Int}) where {T,N} = CUDA.CuArray{T,N}(undef, dims...)
    allocate_array(::AMDGPUBackend, ::Type{T}, dims::NTuple{N,Int}) where {T,N} = AMDGPU.ROCArray{T,N}(undef, dims...)

    ciao() = println("DIOCAN")

    export ciao
    export AbstractBackend, CUDABackend, AMDGPUBackend, CPUBackend
    export BACKEND, set_backend!
    export backend_sin, backend_cos
    export allocate_array

end # module
