###
### "THE BEER-WARE LICENSE":
### Alberto Ramos wrote this file. As long as you retain this 
### notice you can do whatever you want with this stuff. If we meet some 
### day, and you think this stuff is worth it, you can buy me a beer in 
### return. <alberto.ramos@cern.ch>
###
### file:    AlgebraSU2.jl
### created: Sun Oct  3 09:24:25 2021
###                               

SU2alg(x::T)                       where T <: AbstractFloat = SU2alg{T}(x,0.0,0.0)
SU2alg(v::Vector{T})               where T <: AbstractFloat = SU2alg{T}(v[1],v[2],v[3])
projalg(g::SU2{T})                 where T <: AbstractFloat = SU2alg{T}(imag(g.t1), real(g.t2), imag(g.t2))
dot(a::SU2alg{T}, b::SU2alg{T})    where T <: AbstractFloat = a.t1*b.t1 + a.t2*b.t2 + a.t3*b.t3
norm(a::SU2alg{T})             where T <: AbstractFloat = sqrt(a.t1^2 + a.t2^2 + a.t3^2)
norm2(a::SU2alg{T})            where T <: AbstractFloat = a.t1^2 + a.t2^2 + a.t3^2

Base.:+(a::SU2alg{T})              where T <: AbstractFloat = SU2alg{T}(a.t1,a.t2,a.t3)
Base.:-(a::SU2alg{T})              where T <: AbstractFloat = SU2alg{T}(-a.t1,-a.t2,-a.t3)
Base.:+(a::SU2alg{T},b::SU2alg{T}) where T <: AbstractFloat = SU2alg{T}(a.t1+b.t1,a.t2+b.t2,a.t3+b.t3)
Base.:-(a::SU2alg{T},b::SU2alg{T}) where T <: AbstractFloat = SU2alg{T}(a.t1-b.t1,a.t2-b.t2,a.t3-b.t3)

Base.:*(a::SU2alg{T},b::Number)    where T <: AbstractFloat = SU2alg{T}(a.t1*b,a.t2*b,a.t3*b)
Base.:*(b::Number,a::SU2alg{T})    where T <: AbstractFloat = SU2alg{T}(a.t1*b,a.t2*b,a.t3*b)
Base.:/(a::SU2alg{T},b::Number)    where T <: AbstractFloat = SU2alg{T}(a.t1/b,a.t2/b,a.t3/b)

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
