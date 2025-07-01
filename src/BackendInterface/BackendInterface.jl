module BackendInterface
    using CUDA, AMDGPU, Random

    abstract type AbstractBackend end
    struct CPUBackend <: AbstractBackend end
    struct CUDABackend <: AbstractBackend end
    struct AMDGPUBackend <: AbstractBackend end

    const BACKEND = Ref{AbstractBackend}(CPUBackend())

    function set_backend!(b::AbstractBackend)
        BACKEND[] = b
    end
    function set_backend!(backend_symbol::Symbol)
        if backend_symbol == :cpu
            BACKEND[] = CPUBackend()
        elseif backend_symbol == :cuda
            BACKEND[] = CUDABackend()
        elseif backend_symbol == :amd
            BACKEND[] = AMDGPUBackend()
        else
            error("Unknown backend")
        end
    end

    # ---------------- Generic algebra dispatch ------------------
        _sin(::CPUBackend   ,x) = sin(x)       
        _sin(::CUDABackend  ,x) = CUDA.sin(x)  
        _sin(::AMDGPUBackend,x) = AMDGPU.sin(x)
        backend_sin(x) = _sin(BACKEND[],x)

        _cos(::CPUBackend   ,x) = cos(x)       
        _cos(::CUDABackend  ,x) = CUDA.cos(x)  
        _cos(::AMDGPUBackend,x) = AMDGPU.cos(x)
        backend_cos(x) = _cos(BACKEND[],x)
    # ------------------------------------------------------------

    # --------------- Array allocation dispatch -------------------
        _array(::CPUBackend   , ::Type{T}, dims::NTuple{N,Int}) where {T,N} = 
            Array{T,N}(undef, dims...)
        _array(::CUDABackend  , ::Type{T}, dims::NTuple{N,Int}) where {T,N} = 
            CUDA.CuArray{T,N}(undef, dims...)
        _array(::AMDGPUBackend, ::Type{T}, dims::NTuple{N,Int}) where {T,N} = 
            AMDGPU.ROCArray{T,N}(undef, dims...)
        backend_array(::Type{T}, dims::NTuple{N,Int}) where {T,N} = _array(BACKEND[], T, dims)
    # ------------------------------------------------------------

    # --------------- Random number generation --------------------
        _randn(::CPUBackend  ,args...) = Random.randn(args...)
        _randn(::CUDABackend ,args...) = CUDA.randn(args...)  
        _randn(::AMDGPUBackend,args...) = AMDGPU.randn(args...)
        backend_randn(args...) = _randn(BACKEND[], args...)
    # ------------------------------------------------------------


    # ---------------- kernel programming -------------------------
        const CPU_THREAD_IDX = Ref(1)
        const CPU_BLOCK_IDX = Ref(1)
        CPUthreadIdx() = CPU_THREAD_IDX[]
        CPUblockIdx() = CPU_BLOCK_IDX[]

        _kernel(::CPUBackend, kernel!, bsize, gsize, args...) = begin
            for r in 1:gsize
                for b in 1:bsize
                    CPU_THREAD_IDX[] = b
                    CPU_BLOCK_IDX[] = r
                    kernel!(args...)
                end
            end
        end
        _kernel(::CUDABackend, kernel!, bsize, gsize, args...) =  CUDA.@sync begin
            @cuda threads=bsize blocks=gsize kernel!(args...) 
        end
        _kernel(::AMDGPUBackend, kernel!, bsize, gsize, args...) =  begin
            AMDGPU.@roc groupsize=bsize gridsize=gsize kernel!(args...) 
            AMDGPU.synchronize()
        end
        backend_kernel(ker,bsz,gsz,args...) = _kernel(BACKEND[],ker,bsz,gsz,args...)
    # ------------------------------------------------------------
    

    ciao() = println("DIOCAN")

    export ciao
    export AbstractBackend, CUDABackend, AMDGPUBackend, CPUBackend
    export BACKEND, set_backend!
    export backend_sin, backend_cos
    export backend_array, backend_randn
    export backend_kernel, CPUblockIdx, CPUthreadIdx

end # module
