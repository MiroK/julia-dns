# First part of DEBUG

include("utils.jl")
using Utils

N = 16
Nh = N÷2+1

"Indices of k index in the last axis"
view(k::Integer) = (Colon(), Colon(), Colon(), k)

"View of A along the last axis"
function _{T, N}(A::AbstractArray{T, N}, k::Integer)
   @assert 1 <= k <= last(size(A))
   indices = [fill(Colon(), N-1)..., k]
   slice(A, indices...)
end

########################
# 1. dns0 way
########################
# Complex grid
kx = fftfreq(N, 1./N)
kz = kx[1:(N÷2+1)]; kz[end] *= -1
k = ndgrid(kz, kx, kx)
K = Array[k...]
# Square of wave number vectors
K2 = K[1].^2 + K[2].^2 + K[3].^2
# K/K2 term
K_over_K2 = Array[K[i]./K2 for i in 1:3]
# Fix division by zero
for i in 1:3 K_over_K2[i][1] = K[i][1] end
# Dealising mask
kmax_dealias = 2*Nh/3
dealias = (abs(K[1]).<kmax_dealias).*(abs(K[2]).<kmax_dealias).*(abs(K[3]).<kmax_dealias)

########################
# 1. 4d array way
########################
Kd = Array{Float64}(Nh, N, N, 3)
for (i, Ki) in enumerate(k) Kd[view(i)...] = Ki end
# Square of wave number vectors
K2d = sumabs2(Kd, 4)
# K/K2
K_over_K2d = Kd./K2d
# Fix division by zero
for i in 1:3 K_over_K2d[1, 1, 1, i] = Kd[1, 1, 1, i] end
# Dealising mask
dealiasd = reshape(reduce(&, [abs(_(Kd, i)) .< kmax_dealias for i in 1:3]), Nh, N, N, 1)

# Check K
println("K $([maximum(abs(_(Kd, i)-K[i])) for i in 1:3])")
# Check K2
println("K2 $(maximum(abs(K2-K2d)))")
# Check K/K2
println("K/K2 $([maximum(abs(_(K_over_K2d, i)-K_over_K2[i])) for i in 1:3])")
# Check dealias
println("dealias $(maximum(abs(dealias-dealiasd)))")
# Make sure that they were not zeros
println("\t$(map(A->maximum(abs(A)), (Kd, K2d, K_over_K2d, dealiasd)))\n")

########################
# 2.
########################
dUs = (rand(Nh, N, N), rand(Nh, N, N), rand(Nh, N, N))

dU = Array[dUs...]

dUd = similar(Kd)
for (i, dUi) in enumerate(dUs) dUd[view(i)...] = dUi end
# dns0 way
for i in 1:3 dU[i] .*= dealias end
# 4d way
println(eltype(dUd))
dUd .*= dealiasd
println(eltype(dUd))
# Difference
println("dU $([maximum(abs(_(dUd, i)-dU[i])) for i in 1:3])\n")

########################
# 3. FFTS
########################
Xs = rand(16, 16, 16), rand(16, 16, 16), rand(16, 16, 16)

fX = Array{Complex{Float64}}(9, 16, 16)
fY = similar(fX)
fZ = similar(fX)

A = Array[Xs...]
fA = Array[fX, fY, fZ]
ffA = deepcopy(A)

for i in 1:3 fftn_mpi!(A[i], fA[i]) end
for i in 1:3 ifftn_mpi!(fA[i], ffA[i]) end
# Diff
println("FFT-A $([maximum(abs(A[i]-ffA[i])) for i in 1:3])")
# Now let's oragnize the data into one array
B = Array{Float64}(16, 16, 16, 3)
fB = Array{Complex{Float64}}(9, 16, 16, 3) 
for (i, Xi) in enumerate(Xs) B[view(i)...] = Xi end
for i in 1:3 fftn_mpi!(_(B, i), _(fB, i)) end
# See if  fB same as fA
println("fA, fB $([maximum(abs(_(fB, i)-fA[i])) for i in 1:3])")
# Go back
ffB = similar(B)
for i in 1:3 ifftn_mpi!(_(fB, i), _(ffB, i)) end
# See if  ffB same as A
println("A, ffB $([maximum(abs(A[i]-_(ffB, i))) for i in 1:3])\n")

########################
# 4. Cross
########################
function Cross!(a, b, c)
    fftn_mpi!(a[2].*b[3]-a[3].*b[2] , c[1])
    fftn_mpi!(a[3].*b[1]-a[1].*b[3] , c[2])
    fftn_mpi!(a[1].*b[2]-a[2].*b[1] , c[3])
end

function Crossd!(a, b, c)
    fftn_mpi!(_(a, 2).*_(b, 3)-_(a, 3).*_(b, 2), _(c, 1))
    fftn_mpi!(_(a, 3).*_(b, 1)-_(a, 1).*_(b, 3), _(c, 2))
    fftn_mpi!(_(a, 1).*_(b, 2)-_(a, 2).*_(b, 1), _(c, 3))
end

as = rand(16, 16, 16), rand(16, 16, 16), rand(16, 16, 16)
bs = rand(16, 16, 16), rand(16, 16, 16), rand(16, 16, 16)

a, b = Array[as...], Array[bs...]

ad = Array{Float64}(16, 16, 16, 3)
bd = Array{Float64}(16, 16, 16, 3) 
for (i, ai) in enumerate(as) ad[view(i)...] = ai end
for (i, bi) in enumerate(bs) bd[view(i)...] = bi end

cd = Array{Complex{Float64}}(9, 16, 16, 3) 

cs = Array{Complex{Float64}}(9, 16, 16), Array{Complex{Float64}}(9, 16, 16), Array{Complex{Float64}}(9, 16, 16)
c = Array[cs...]

Cross!(a, b, c)
Crossd!(ad, bd, cd)
# Error
println("cross $([maximum(abs(c[i]-_(cd, i))) for i in 1:3])")
println("\t$(sumabs2(cd))\n")

########################
# 4. Curl
########################
function Curl!(a, K, c)
    ifftn_mpi!(1.0im*(K[1].*a[2]-K[2].*a[1]), c[3])
    ifftn_mpi!(1.0im*(K[3].*a[1]-K[1].*a[3]), c[2])
    ifftn_mpi!(1.0im*(K[2].*a[3]-K[3].*a[2]), c[1])
end

function Curld!(a, K, c)
    ifftn_mpi!(im*(_(K, 1).*_(a, 2)-_(K, 2).*_(a, 1)), _(c, 3))
    ifftn_mpi!(im*(_(K, 3).*_(a, 1)-_(K, 1).*_(a, 3)), _(c, 2))
    ifftn_mpi!(im*(_(K, 2).*_(a, 3)-_(K, 3).*_(a, 2)), _(c, 1))
end

Curl!(dU, K, a)
Curld!(dUd, Kd, ad)
println("curl $([maximum(abs(a[i]-_(ad, i))) for i in 1:3])")
println("\t$(sumabs2(ad))\n")

########################
# 5.
########################
ys = rand(Complex128, Nh, N, N), rand(Complex128, Nh, N, N), rand(Complex128, Nh, N, N)
Uhat = Array[ys...]

Uhatd = Array{Complex128}(Nh, N, N, 3)
for (i, yi) in enumerate(ys) Uhatd[view(i)...] = yi end

d3 = [K2.*Uhat[i] for i in 1:3]
d4 = K2d.*Uhatd
println(":: $([sumabs2(d3[i]-_(d4, i)) for i in 1:3])")
