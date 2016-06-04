type SpecTransf{T<:Real}
    plan::FFTW.rFFTWPlan{T, -1, false, 3} 
    inv_plan
    function SpecTransf(A)
        sizes = size(A)
        N = first(sizes)
        Nh = N÷2+1

        fA = Array{Complex{T}}(Nh, N, N)
        new(plan_rfft(A, (1, 2, 3)), plan_irfft(fA, N, (1, 2, 3)))
    end
end
SpecTransf{T<:Real}(A::AbstractArray{T, 3}) = SpecTransf{T}(A)

function apply{T}(fu::AbstractArray{Complex{T}, 3},
                  f::SpecTransf{T},
                  u::AbstractArray{T, 3})
    A_mul_B!(fu, f.plan, u)
end

function apply_inv{T}(u::AbstractArray{T, 3},
                      f::SpecTransf{T},
                      fu::AbstractArray{Complex{T}, 3})
    A_mul_B!(u, f.inv_plan, fu)
end

A = rand(128, 128, 128)
const f = SpecTransf(A)
const F = plan_rfft(A, (1, 2, 3))

real2spec{T}(u::AbstractArray{T, 3},
             fu::AbstractArray{Complex{T}, 3}) = apply(fu, f, u)

spec2real{T}(fu::AbstractArray{Complex{T}, 3},
             u::AbstractArray{T, 3}) = apply_inv(u, f, fu)

fftn_mpi!(u, fu) = A_mul_B!(fu, F, u)


function foo(A, n=1)
    FA = similar(rfft(A, (1, 2, 3)))
    @time for i in 1:n fftn_mpi!(A, FA) end
end

function bar(A, n=1)
    FA = similar(rfft(A, (1, 2, 3)))
    @time for i in 1:n real2spec(A, FA) end
    
    B = similar(A)
    spec2real(FA, B)
    sumabs2(B-A)
end

foo(A)
bar(A)


foo(A, 100)

bar(A, 100)
