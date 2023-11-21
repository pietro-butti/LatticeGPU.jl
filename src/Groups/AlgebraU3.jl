


struct U3alg{T} <: Algebra
    u11::T
    u22::T
    u33::T
    u12::Complex{T}
    u13::Complex{T}
    u23::Complex{T}
end

function antsym(a::SU3{T}) where T <: AbstractFloat
    t1 = 2.0*imag(a.u11)
    t2 = 2.0*imag(a.u22)
    t3 = 2.0*imag(a.u12*a.u21 - a.u11*a.u22)
    t4 = a.u12 - conj(a.u21)
    t5 = a.u13 - (a.u12*a.u23 - a.u13*a.u22)
    t6 = a.u23 - (a.u13*a.u21 - a.u11*a.u23)

    return U3alg{T}(t1,t2,t3,t4,t5,t6)
end


Base.:*(a::U3alg{T},b::SU3fund{T}) where T <: AbstractFloat = SU3fund{T}(im*a.u11*b.t1 + a.u12*b.t2 + a.u13*b.t3,
                                                                        -conj(a.u12)*b.t1 + im*a.u22*b.t2 + a.u23*b.t3,
                                                                        -conj(a.u13)*b.t1 - conj(a.u23)*b.t2 + im*a.u33*b.t3)

Base.:+(a::U3alg{T},b::U3alg{T}) where T <: AbstractFloat = U3alg{T}(a.u11 + b.u11, a.u22 + b.u22, a.u33 + b.u33, a.u12 + b.u12, a.u13 + b.u13, a.u23 + b.u23)


Base.:*(r::Number, a::U3alg{T}) where T <: AbstractFloat = U3alg{T}(r*a.u11, r*a.u22, r*a.u33, r*a.u12, r*a.u13, r*a.u23)

