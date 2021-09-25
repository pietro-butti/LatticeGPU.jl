###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    GroupSU2.jl
### created: Sun Jul 11 17:23:12 2021
###                               

#
# SU(2) group elements represented trough Cayley-Dickson
#       construction
# https://en.wikipedia.org/wiki/Cayley%E2%80%93Dickson_construction
using CUDA, Random

import Base.:*, Base.:+, Base.:-,Base.:/,Base.:\,Base.exp,Base.zero,Base.one
import Random.rand
struct SU2{T} <: Group
    t1::Complex{T}
    t2::Complex{T}
end
SU2(a::T, b::T)          where T <: AbstractFloat = SU2{T}(complex(a), complex(b))
inverse(b::SU2{T})       where T <: AbstractFloat = SU2{T}(conj(b.t1), -b.t2)
dag(a::SU2{T})           where T <: AbstractFloat = inverse(a)
norm(a::SU2{T})          where T <: AbstractFloat = sqrt(abs2(a.t1) + abs2(a.t2))
norm2(a::SU2{T})         where T <: AbstractFloat = abs2(a.t1) + abs2(a.t2)
tr(g::SU2{T})            where T <: AbstractFloat = complex(2.0*real(g.t1), 0.0)
Base.one(::Type{SU2{T}}) where T <: AbstractFloat = SU2{T}(one(T),zero(T))
Random.rand(rng::AbstractRNG, ::Random.SamplerType{SU2{T}}) where T <: AbstractFloat = exp(SU2alg{T}(randn(rng,T),randn(rng,T),randn(rng,T)))

"""
    function normalize(a::T) where {T <: Group}

Return a normalized element of the group.
"""
function normalize(a::SU2{T}) where T <: AbstractFloat
    dr = sqrt(abs2(a.t1) + abs2(a.t2))
    if (dr == 0.0)
        return SU2(0.0)
    end
    return SU2{T}(a.t1/dr,a.t2/dr)
end

Base.:*(a::SU2{T},b::SU2{T}) where T <: AbstractFloat = SU2{T}(a.t1*b.t1-a.t2*conj(b.t2),a.t1*b.t2+a.t2*conj(b.t1))
Base.:/(a::SU2{T},b::SU2{T}) where T <: AbstractFloat = SU2{T}(a.t1*conj(b.t1)+a.t2*conj(b.t2),-a.t1*b.t2+a.t2*b.t1)
Base.:\(a::SU2{T},b::SU2{T}) where T <: AbstractFloat = SU2{T}(conj(a.t1)*b.t1+a.t2*conj(b.t2),conj(a.t1)*b.t2-a.t2*conj(b.t1))

struct SU2alg{T} <: Algebra
    t1::T
    t2::T
    t3::T
end
SU2alg(x::T)                       where T <: AbstractFloat = SU2alg{T}(x,0.0,0.0)
SU2alg(v::Vector{T})               where T <: AbstractFloat = SU2alg{T}(v[1],v[2],v[3])
projalg(g::SU2{T})                 where T <: AbstractFloat = SU2alg{T}(imag(g.t1), real(g.t2), imag(g.t2))
dot(a::SU2alg{T}, b::SU2alg{T})    where T <: AbstractFloat = a.t1*b.t1 + a.t2*b.t2 + a.t3*b.t3
norm(a::SU2alg{T})             where T <: AbstractFloat = sqrt(a.t1^2 + a.t2^2 + a.t3^2)
norm2(a::SU2alg{T})            where T <: AbstractFloat = a.t1^2 + a.t2^2 + a.t3^2
Base.zero(::Type{SU2alg{T}})      where T <: AbstractFloat = SU2alg{T}(zero(T),zero(T),zero(T))
Random.rand(rng::AbstractRNG, ::Random.SamplerType{SU2alg{T}}) where T <: AbstractFloat = SU2alg{T}(randn(rng,T),randn(rng,T),randn(rng,T))

Base.:+(a::SU2alg{T})              where T <: AbstractFloat = SU2alg{T}(a.t1,a.t2,a.t3)
Base.:-(a::SU2alg{T})              where T <: AbstractFloat = SU2alg{T}(-a.t1,-a.t2,-a.t3)
Base.:+(a::SU2alg{T},b::SU2alg{T}) where T <: AbstractFloat = SU2alg{T}(a.t1+b.t1,a.t2+b.t2,a.t3+b.t3)
Base.:-(a::SU2alg{T},b::SU2alg{T}) where T <: AbstractFloat = SU2alg{T}(a.t1-b.t1,a.t2-b.t2,a.t3-b.t3)

Base.:*(a::SU2alg{T},b::Number)    where T <: AbstractFloat = SU2alg{T}(a.t1*b,a.t2*b,a.t3*b)
Base.:*(b::Number,a::SU2alg{T})    where T <: AbstractFloat = SU2alg{T}(a.t1*b,a.t2*b,a.t3*b)
Base.:/(a::SU2alg{T},b::Number)    where T <: AbstractFloat = SU2alg{T}(a.t1/b,a.t2/b,a.t3/b)

function isgroup(a::SU2{T}) where T <: AbstractFloat
    tol = 1.0E-10
    if (abs2(a.t1) + abs2(a.t2) - 1.0 < 1.0E-10)
        return true
    else
        return false
    end
end

"""
    function Base.exp(a::T, t::Number=1) where {T <: Algebra}

Computes `exp(a)`
"""
function Base.exp(a::SU2alg{T}) where T <: AbstractFloat
    
    rm = sqrt( a.t1^2+a.t2^2+a.t3^2 )/2.0
    if (abs(rm) < 0.05)
        rms = rm^2/2.0
        ca = 1.0 - rms    *(1.0 - (rms/6.0 )*(1.0 - rms/15.0))
        sa = 0.5 - rms/6.0*(1.0 - (rms/10.0)*(1.0 - rms/21.0))
    else
        ca = CUDA.cos(rm)
	sa = CUDA.sin(rm)/(2.0*rm)
    end

    t1 = complex(ca,sa*a.t1)
    t2 = complex(sa*a.t2,sa*a.t3)
    return SU2{T}(t1,t2)
end

function Base.exp(a::SU2alg{T}, t::T) where T <: AbstractFloat
    
    rm = t*sqrt( a.t1^2+a.t2^2+a.t3^2 )/2.0
    if (abs(rm) < 0.05)
        rms = rm^2/2.0
        ca = 1.0 - rms    *(1.0 - (rms/6.0 )*(1.0 - rms/15.0))
        sa = t*(0.5 - rms/6.0*(1.0 - (rms/10.0)*(1.0 - rms/21.0)))
    else
        ca = CUDA.cos(rm)
	sa = t*CUDA.sin(rm)/(2.0*rm)
    end

    t1 = complex(ca,sa*a.t1)
    t2 = complex(sa*a.t2,sa*a.t3)
    return SU2{T}(t1,t2)
end


"""
    function expm(g::G, a::A) where {G <: Algebra, A <: Algebra}

Computes `exp(a)*g`

"""
function expm(g::SU2{T}, a::SU2alg{T}) where T <: AbstractFloat
    
    rm = sqrt( a.t1^2+a.t2^2+a.t3^2 )/2.0
    if (abs(rm) < 0.05)
        rms = rm^2/2.0
        ca = 1.0 - rms    *(1.0 - (rms/6.0 )*(1.0 - rms/15.0))
        sa = 0.5 - rms/6.0*(1.0 - (rms/10.0)*(1.0 - rms/21.0))
    else
        ca = CUDA.cos(rm)
	sa = CUDA.sin(rm)/(2.0*rm)
    end

    t1 = complex(ca,sa*a.t1)*g.t1-complex(sa*a.t2,sa*a.t3)*conj(g.t2)
    t2 = complex(ca,sa*a.t1)*g.t2+complex(sa*a.t2,sa*a.t3)*conj(g.t1)
    return SU2{T}(t1,t2)
end

"""
    function expm(g::SU2, a::SU2alg, t::Float64)

Computes `exp(t*a)*g`

"""
function expm(g::SU2{T}, a::SU2alg{T}, t::T) where T <: AbstractFloat
    
    rm = t*sqrt( a.t1^2+a.t2^2+a.t3^2 )/2.0
    if (abs(rm) < 0.05)
        rms = rm^2/2.0
        ca = 1.0 - rms    *(1.0 - (rms/6.0 )*(1.0 - rms/15.0))
        sa = t*(0.5 - rms/6.0*(1.0 - (rms/10.0)*(1.0 - rms/21.0)))
    else
        ca = CUDA.cos(rm)
	sa = t*CUDA.sin(rm)/(2.0*rm)
    end

    t1 = complex(ca,sa*a.t1)*g.t1-complex(sa*a.t2,sa*a.t3)*conj(g.t2)
    t2 = complex(ca,sa*a.t1)*g.t2+complex(sa*a.t2,sa*a.t3)*conj(g.t1)
    return SU2{T}(t1,t2)
               
end

export SU2, SU2alg, inverse, dag, tr, projalg, expm, exp, norm, norm2, isgroup
