###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    YMfields.jl
### created: Thu Jul 15 15:16:47 2021
###                               


using AMDGPU

"""
        function randomize!(f, lp::SpaceParm, ymws::YMworkspace)

Given an algebra field with natural indexing, this routine sets the components to random Gaussian distributed values. If SF boundary conditions are used, the force at the boundaries is set to zero.
"""
function randomize!(f, lp::SpaceParm, ymws::YMworkspace) 
    m = backend_randn(ymws.PRC, lp.bsz,lp.ndim,3,lp.rsz)
    begin
        AMDGPU.@roc groupsize=lp.bsz gridsize=lp.rsz krnl_assign_SU2_amd!(f,m,lp) 
        AMDGPU.synchronize()
    end    



    # if ymws.ALG == SU2alg
    #     @timeit "Randomize SU(2) algebra field" begin
    #         m = backend_randn(ymws.PRC, lp.bsz,lp.ndim,3,lp.rsz)
    #         BACKEND[] isa CPUBackend    ? backend_kernel(krnl_assign_SU2_cpu! , lp.bsz, lp.rsz, f, m, lp) :
    #         BACKEND[] isa CUDABackend   ? backend_kernel(krnl_assign_SU2_cuda!, lp.bsz, lp.rsz, f, m, lp) :
    #         # BACKEND[] isa AMDGPUBackend ? backend_kernel(krnl_assign_SU2_amd! , lp.bsz, lp.rsz, f, m, lp) :
    #         BACKEND[] isa AMDGPUBackend ? backend_kernel(krnl! , lp.bsz, lp.rsz, f, m, lp) :
    #         error("Backend not supported")
    #     end
    #     return nothing
    # end

    # if ymws.ALG == SU3alg
    #     @timeit "Randomize SU(3) algebra field" begin
    #         m = backend_randn(ymws.PRC, lp.bsz,lp.ndim,8,lp.rsz)
    #         BACKEND[] isa CPUBackend    ? backend_kernel(krnl_assign_SU3_cpu! , lp.bsz, lp.rsz, f, m, lp) :
    #         BACKEND[] isa CUDABackend   ? backend_kernel(krnl_assign_SU3_cuda!, lp.bsz, lp.rsz, f, m, lp) :
    #         BACKEND[] isa AMDGPUBackend ? backend_kernel(krnl_assign_SU3_amd! , lp.bsz, lp.rsz, f, m, lp) :
    #         error("Backend not supported")
    #     end
    #     return nothing
    # end

    return nothing
end




function make_krnl_assign_SU2!(name::Symbol, backend::Symbol, threadIdx_expr::Expr, blockIdx_expr::Expr)
    quote
        function $(name)(frc, m, lp::SpaceParm{N,M,BC_PERIODIC,D}) where {N,M,D}
            @inbounds begin
                # backend-specific part
                b = Int($threadIdx_expr)
                r = Int($blockIdx_expr )
       
                # common body part
                for id in 1:lp.ndim
                    frc[b,id,r] = SU2alg(m[b,id,1,r], m[b,id,2,r], m[b,id,3,r])
                end
            end

            return nothing
        end
    end
end
@eval $(make_krnl_assign_SU2!(
    :krnl_assign_SU2_cpu!, :CPUBackend, 
    :(CPUthreadIdx()), :(CPUblockIdx())
)) 
@eval $(make_krnl_assign_SU2!(
    :krnl_assign_SU2_cuda!,:CUDABackend  , 
    :(CUDA.threadIdx().x), :(CUDA.blockIdx().x)
)) 
@eval $(make_krnl_assign_SU2!(
    :krnl_assign_SU2_amd! ,:AMDGPUBackend, 
    :(AMDGPU.Device.threadIdx().x), :(AMDGPU.Device.blockIdx().x)
))




# function make_krnl_assign_SU3!(name::Symbol, backend::Symbol, threadIdx_expr::Expr, blockIdx_expr::Expr)
#     quote

#         function $(name)(frc::AbstractArray{T}, m, lp::SpaceParm{N,M,BC_PERIODIC,D}) where {T,N,M,D}

#             @inbounds begin
#                 b = $threadIdx_expr |> Int
#                 r = $blockIdx_expr |> Int
#                 for id in 1:lp.ndim
#                     frc[b,id,r] = SU3alg(m[b,id,1,r], m[b,id,2,r], m[b,id,3,r],
#                                         m[b,id,4,r], m[b,id,5,r], m[b,id,6,r],
#                                         m[b,id,7,r], m[b,id,8,r])
#                 end
#             end
            
#             return nothing
#         end

#         function $(name)(frc::AbstractArray{T}, m, lp::SpaceParm{N,M,BC_OPEN,D}) where {T,N,M,D}

#             @inbounds begin
#                 b = $threadIdx_expr |> Int
#                 r = $blockIdx_expr |> Int
#                 for id in 1:lp.ndim
#                     frc[b,id,r] = SU3alg(m[b,id,1,r], m[b,id,2,r], m[b,id,3,r],
#                                         m[b,id,4,r], m[b,id,5,r], m[b,id,6,r],
#                                         m[b,id,7,r], m[b,id,8,r])
#                 end
#             end

#             return nothing
#         end

#         function $(name)(frc::AbstractArray{T}, m, lp::Union{SpaceParm{N,M,BC_SF_ORBI,D},SpaceParm{N,M,BC_SF_AFWB,D}}) where {T,N,M,D}

#             @inbounds begin
#                 b = $threadIdx_expr |> Int
#                 r = $blockIdx_expr |> Int
#                 it = point_time((b,r), lp)

#                 if it == 1
#                     for id in 1:lp.ndim-1
#                         frc[b,id,r] = zero(T)
#                     end
#                     frc[b,N,r] = SU3alg(m[b,N,1,r], m[b,N,2,r], m[b,N,3,r],
#                                         m[b,N,4,r], m[b,N,5,r], m[b,N,6,r],
#                                         m[b,N,7,r], m[b,N,8,r])
#                 else
#                     for id in 1:lp.ndim
#                         frc[b,id,r] = SU3alg(m[b,id,1,r], m[b,id,2,r], m[b,id,3,r],
#                                             m[b,id,4,r], m[b,id,5,r], m[b,id,6,r],
#                                             m[b,id,7,r], m[b,id,8,r])
#                     end
#                 end
#             end

#             return nothing
#         end

#     end
# end
# @eval $(make_krnl_assign_SU3!(
#     :krnl_assign_SU3_cpu!, :CPUBackend, 
#     :(CPUthreadIdx()), :(CPUblockIdx())
# )) 
# @eval $(make_krnl_assign_SU3!(
#     :krnl_assign_SU3_cuda!,:CUDABackend  , 
#     :(CUDA.threadIdx().x), :(CUDA.blockIdx().x)
# )) 
# @eval $(make_krnl_assign_SU3!(
#     :krnl_assign_SU3_amd! ,:AMDGPUBackend, 
#     :(AMDGPU.Device.threadIdx().x), :(AMDGPU.Device.blockIdx().x)
# ))








