# LatticeGPU.jl: A framework for lattice computations on the GPU

This is a fork of the [original LatticeGPU.jl repository][https://igit.ific.uv.es/alramos/latticegpu.jl.git]

The main goal is to extend `LatticeGPU` beyond `CUDA.jl`.


## Methodology followed:
- For every "GPU" backend (CUDA, AMD, CPU) we define a subtype of `AbstractBackend`
- A global constant `BACKEND` select which backend to use (can be changed by the user by calling `set_backend!`)

### Simple functions (arithmetics and arrays allocation) 
- Every relelvant function is dispatched for every specific backend by taking as first argument an `AbstractBackend` subtype. E.g.
```
_fun(::CPUBackend   ,x) =        fun(x)       
_fun(::CUDABackend  ,x) =   CUDA.fun(x)  
_fun(::AMDGPUBackend,x) = AMDGPU.fun(x)
backend_fun(x) = _fun(BACKEND[],x)
export backend_fun
```
- In the original code, every call to `CUDA.fun` is replaced by `backend_fun`

#### Some design choices:
- all relevant function `<func name>` that needs to be dispatched is named with an underscore: `_<func name>`, then `backend_<func name>` will be the exported one
(This might help some meta-programming automation for the future)

#### Some doubts:
- is it necessary to give the full signature? Can it just be?
```
_fun(::CPUBackend   ) =        fun       
_fun(::CUDABackend  ) =   CUDA.fun  
_fun(::AMDGPUBackend) = AMDGPU.fun
backend_fun = _fun(BACKEND[])
```

### Kernel and others
- the function that calls the kernels can be dispatched as above, 
- A different kernel has to be written for every backend



## Log of changes

- I installed `AMDGPU.jl` simply by adding it with the package manager.

- I added a whole new submodule `BackendInterface.jl`:

- I modified `Groups.jl`:
    - Removed `using CUDA`
    - `CUDA.sin` and `CUDA.cos` replaced with `backend_sin` `backend_cos`

- I modified `Fields.jl`:
    - Removed `using CUDA`
    - Functions defining fields are replaced with a proper call to `backend_array`



## Concerns
- `backend_*` functions are redefined to 