

Base.zero(::Type{SU2fund{T}}) where T <: AbstractFloat = SU2fund{T}(zero(T),zero(T))
Random.rand(rng::AbstractRNG, ::Random.SamplerType{SU2fund{T}}) where T <: AbstractFloat = SU2fund{T}(complex(randn(rng,T),randn(rng,T)),
                                                                                                      complex(randn(rng,T),randn(rng,T)))


SU2fund(a::T, b::T)          where T <: AbstractFloat = SU2fund{T}(complex(a), complex(b))

"""
    dag(a::SU2fund{T})

Returns the conjugate of a fundamental element. 
"""
dag(a::SU2fund{T})                 where T <: AbstractFloat = SU2fund{T}(conj(a.t1), conj(a.t2))

"""
    norm(a::SU2fund{T})

Returns the norm of a fundamental element. Same result as dot(a,a).
"""
norm(a::SU2fund{T})                where T <: AbstractFloat = sqrt((abs2(a.t1) + abs2(a.t2)))

"""
    norm(a::SU2fund{T})

Returns the norm of a fundamental element. Same result as sqrt(dot(a,a)).
"""
norm2(a::SU2fund{T})               where T <: AbstractFloat = (abs2(a.t1) + abs2(a.t2))

"""
    dot(a::SU2fund{T},b::SU2fund{T})

Returns the scalar product of two fundamental elements. The convention is for the product to the linear in the second argument, and anti-linear in the first argument. 
"""
dot(g1::SU2fund{T},g2::SU2fund{T}) where T <: AbstractFloat = conj(g1.t1)*g2.t1+conj(g1.t2)*g2.t2

"""
    *(g::SU2{T},b::SU2fund{T})

Returns ga
"""
Base.:*(g::SU2{T},b::SU2fund{T})     where T <: AbstractFloat = SU2fund{T}(g.t1*b.t1 + g.t2*b.t2,-conj(g.t2)*b.t1 + conj(g.t1)*b.t2)
                                                                           
"""
    \\(g::SU2{T},b::SU2fund{T})

Returns g^dag b
"""
Base.:\(g::SU2{T},b::SU2fund{T})     where T <: AbstractFloat = SU2fund{T}(conj(g.t1)*b.t1 - g.t2 * b.t2,conj(g.t2)*b.t1 + g.t1 * b.t2)

"""
    *(a::SU2alg{T},b::SU2fund{T})

Returns a*b
"""

Base.:*(a::SU2alg{T},b::SU2fund{T})     where T <: AbstractFloat = SU2fund{T}(complex(0.0, a.t3)*b.t1/2 +  complex(a.t2,a.t1)*b.t2/2,
                                                                              complex(-a.t2,a.t1)*b.t1/2 + complex(0.0, -a.t3)*b.t1/2)

Base.:+(a::SU2fund{T},b::SU2fund{T}) where T <: AbstractFloat = SU2fund{T}(a.t1+b.t1,a.t2+b.t2)
Base.:-(a::SU2fund{T},b::SU2fund{T}) where T <: AbstractFloat = SU2fund{T}(a.t1-b.t1,a.t2-b.t2)
Base.:+(a::SU2fund{T})               where T <: AbstractFloat = SU2fund{T}(a.t1,a.t2)
Base.:-(a::SU2fund{T})               where T <: AbstractFloat = SU2fund{T}(-a.t1,-a.t2)
imm(a::SU2fund{T})                   where T <: AbstractFloat = SU2fund{T}(complex(-imag(a.t1),real(a.t1)),
                                                                           complex(-imag(a.t2),real(a.t2)))
mimm(a::SU2fund{T})                  where T <: AbstractFloat = SU2fund{T}(complex(imag(a.t1),-real(a.t1)),
                                                                           complex(imag(a.t2),-real(a.t2)))

# Operations with numbers
Base.:*(a::SU2fund{T},b::Number) where T <: AbstractFloat = SU2fund{T}(b*a.t1,b*a.t2)
Base.:*(b::Number,a::SU2fund{T}) where T <: AbstractFloat = SU2fund{T}(b*a.t1,b*a.t2)
Base.:/(a::SU2fund{T},b::Number) where T <: AbstractFloat = SU2fund{T}(a.t1/b,a.t2/b)

Base.:*(a::M2x2{T},b::SU2fund{T}) where T <: AbstractFloat = SU2fund{T}(a.u11*b.t1 + a.u12*b.t2,
                                                                        a.u21*b.t1 + a.u22*b.t2)