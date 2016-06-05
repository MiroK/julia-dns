# Take the example of computing P_hat. We do it with linear
# indexing can it be done faster?

"Linear indexing along last axis"
function linind{T, N}(A::AbstractArray{T, N})
    L = prod(size(A)[1:N-1])
    indices = [1]
    for k in 1:size(A, N) push!(indices, last(indices)+L) end
    indices
end

function foo(dU, K_over_K2, P_hat)
    P_hat[:] = 0im
    indices = linind(dU)
    for axis in 1:last(size(dU))
        for (j, i) in enumerate(indices[axis]:indices[axis+1]-1)
            @inbounds P_hat[j] += dU[i]*K_over_K2[i]
        end
    end
end

# ----------------------------------------------------------------------------

function bar(dU, K_over_K2, P_hat)
    P_hat[:] = 0im
    for i4 in 1:size(dU, 4)
       for i3 in 1:size(dU, 3)
           for i2 in 1:size(dU, 2)
               for i1 in 1:size(dU, 1)
                   @inbounds P_hat[i1, i2, i3] += dU[i1, i2, i3, i4]*K_over_K2[i1, i2, i3, i4]
               end
           end
       end
    end
end

# ----------------------------------------------------------------------------

using Base.Cartesian

@generated function foobar{T, N, M}(dU::Array{Complex{T}, N}, 
                                    K_over_K2::Array{T, N},
                                    P_hat::Array{Complex{T}, M})
    quote
        P_hat[:] = zero(Complex{T})
        @assert M == N-1
        @nloops $N i dU begin
            @inbounds (@nref $M P_hat i) += (@nref $N dU i) * (@nref $N K_over_K2 i)
        end
    end
end

# ----------------------------------------------------------------------------

N = 2^7
Nh = N÷2 + 1
dU = rand(Complex128, Nh, N, N, 3)
K_over_K2 = rand(Float64, Nh, N, N, 3)

P_hat1 = Array{Complex128}(Nh, N, N)
foo(dU, K_over_K2, P_hat1)
@ time for i in 1:10
    foo(dU, K_over_K2, P_hat1)
end
println()

P_hat3 = similar(P_hat1)
foobar(dU, K_over_K2, P_hat3)
@time for i in 1:10
    foobar(dU, K_over_K2, P_hat3)
end
println("$(sumabs2(P_hat1-P_hat3))\n")

P_hat2 = similar(P_hat1)
bar(dU, K_over_K2, P_hat2)
@time for i in 1:10
    bar(dU, K_over_K2, P_hat2)
end
println("$(sumabs2(P_hat1-P_hat2))\n")
