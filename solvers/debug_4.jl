include("utils.jl")
using Utils

"Indices of k index in the last axis"
view(k::Integer) = (Colon(), Colon(), Colon(), k)

"View of A along the last axis"
function _{T, N}(A::AbstractArray{T, N}, k::Integer)
   @assert 1 <= k <= last(size(A))
   indices = [fill(Colon(), N-1)..., k]
   slice(A, indices...)
end

const N = 16
const Nh = N÷2+1
const dt = 0.01

# Real vectors
U    = Array{Float64}(N, N, N, 3)
curl = similar(U)
# Complex vectors
dU     = Array{Complex128}(Nh, N, N, 3)
U_hat  = similar(dU) 
# Complex scalars
P_hat  = Array{Complex128}(Nh, N, N)
# Real scalars
P = Array{Float64}(N, N, N)
# Real grid
x = collect(0:N-1)*2*pi/N
X = similar(U)
for (i, Xi) in enumerate(ndgrid(x, x, x)) X[view(i)...] = Xi end
# Wave number grid
kx = fftfreq(N, 1./N)
kz = kx[1:(N÷2+1)]; kz[end] *= -1
K = Array{Float64}(Nh, N, N, 3)
for (i, Ki) in enumerate(ndgrid(kz, kx, kx)) K[view(i)...] = Ki end
# Square of wave number vectors
K2 = sumabs2(K, 4)
# K/K2
K_over_K2 = K./K2
# Fix division by zero
for i in 1:3 K_over_K2[1, 1, 1, i] = K[1, 1, 1, i] end
# Dealising mask
const kmax_dealias = 2*Nh/3
dealias = reshape(reduce(&, [abs(_(K, i)) .< kmax_dealias for i in 1:3]), Nh, N, N, 1)
a = dt*[1./6., 1./3., 1./3., 1./6.]  
b = dt*[0.5, 0.5, 1.] 

function Cross!(a, b, c)
    fftn_mpi!(_(a, 2).*_(b, 3)-_(a, 3).*_(b, 2), _(c, 1))
    fftn_mpi!(_(a, 3).*_(b, 1)-_(a, 1).*_(b, 3), _(c, 2))
    fftn_mpi!(_(a, 1).*_(b, 2)-_(a, 2).*_(b, 1), _(c, 3))
end

function Curl!(a, K, c)
    ifftn_mpi!(im*(_(K, 1).*_(a, 2)-_(K, 2).*_(a, 1)), _(c, 3))
    ifftn_mpi!(im*(_(K, 3).*_(a, 1)-_(K, 1).*_(a, 3)), _(c, 2))
    ifftn_mpi!(im*(_(K, 2).*_(a, 3)-_(K, 3).*_(a, 2)), _(c, 1))
end

"sources, rk, out"
function ComputeRHS!(U, U_hat, curl, K, K_over_K2, K2, P_hat, nu, rk, dU)
    if rk > 1
        for i in 1:3 ifftn_mpi!(_(U_hat, i), _(U, i)) end
    end

    Curl!(U_hat, K, curl)
    Cross!(U, curl, dU)

    println("dU $([sumabs2(_(dU, i)) for i in 1:3]), $(eltype(dU))")
    
    dU .*= dealias

    println("dU $([sumabs2(_(dU, i)) for i in 1:3]), $(eltype(dU))")

    P_hat[:] = sum(dU.*K_over_K2, 4)
    println("P $(sumabs2(P_hat)), $([sumabs2(_(U_hat, i)) for i in 1:3])")

    dU -= P_hat.*K
    println("dU $([sumabs2(_(dU, i)) for i in 1:3]), $(eltype(dU))")
    dU -= nu*K2.*U_hat   # <----- this seems a problem
end

U[view(1)...] = sin(_(X, 1)).*cos(_(X, 2)).*cos(_(X, 3))
U[view(2)...] = -cos(_(X, 1)).*sin(_(X, 2)).*cos(_(X, 3))
U[view(3)...] = 0.

for i in 1:3 fftn_mpi!(_(U, i), _(U_hat, i)) end

rk = 1
nu = 0.000625

println("dU $([sumabs2(_(dU, i)) for i in 1:3])")
println("curl $([sumabs2(_(curl, i)) for i in 1:3])")
println("U $([sumabs2(_(U, i)) for i in 1:3])")
println("P $(sumabs2(P_hat))\n")

for i in 1:3 fftn_mpi!(_(U, i), _(U_hat, i)) end
ComputeRHS!(U, U_hat, curl, K, K_over_K2, K2, P_hat, nu, rk, dU)

println("dU $([sumabs2(_(dU, i)) for i in 1:3])")
println("curl $([sumabs2(_(curl, i)) for i in 1:3])")
println("U $([sumabs2(_(U, i)) for i in 1:3])")
println("P $(sumabs2(P_hat))\n")
