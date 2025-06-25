# LatticeGPU.jl: A framework for lattice computations on the GPU

This is a fork of the [original LatticeGPU.jl repository][https://igit.ific.uv.es/alramos/latticegpu.jl.git]


## Log of changes
The main goal is to extend `LatticeGPU` beyond `CUDA.jl`.

- I installed `AMDGPU.jl` simply by adding it with the package manager.

- I added a whole new submodule `BackendInterface.jl`:

    - Every "GPU type" (CUDA, AMD, CPU) is defined as a subtype of `AbstractBackend` abs. type
    - A global constant `BACKEND` select which backend to use (can be changed through `set_backend!`)
    - All relevant operation are dispatched to multiple types.

- I modified `Goups.jl`:
    - Removed `using CUDA`
    - `CUDA.sin` and `CUDA.cos` replaced with `backend_sin` `backend_cos`

- I modified `Fields.jl`:
    - Removed `using CUDA`
    - Functions defining fields are replaced with a proper call to `BackendInterface.allocate_array`, which takes as first argument `BACKEND[]`
