


"""
    struct U2alg{T} <: Algebra

Elements of the `U(2)` Algebra. The type `T <: AbstractFloat` can be used to define single or double precision elements.
"""
struct U2alg{T} <: Algebra
    u11::T
    u22::T
    u12::Complex{T}
end


"""
    antsym(a::SU2{T}) where T <: AbstractFloat

Returns the antisymmetrization of the SU2 element `a`, that is `\`\ a - a^{\\dagger} `\`. This method returns al element of `U2alg{T}`.
"""
function antsym(a::SU2{T}) where T <: AbstractFloat
    return U2alg{T}(2.0*imag(a.t1),-2.0*imag(a.t1),2.0*a.t2)
end

Base.:*(a::U2alg{T},b::SU2fund{T}) where T <: AbstractFloat = SU2fund{T}(im*a.u11*b.t1 + a.u12*b.t2,
                                                                        -conj(a.u12)*b.t1 + im*a.u22*b.t2)

Base.:+(a::U2alg{T},b::U2alg{T}) where T <: AbstractFloat = U2alg{T}(a.u11 + b.u11, a.u22 + b.u22, a.u12 + b.u12)

Base.:*(r::Number, a::U2alg{T}) where T <: AbstractFloat = U2alg{T}(r*a.u11, r*a.u22, r*a.u12)

