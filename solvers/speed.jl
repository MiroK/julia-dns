include("utils.jl")
using Utils
"View of A along the last axis"
function _{T, N}(A::AbstractArray{T, N}, k::Integer)
   # @assert 1 <= k <= last(size(A))
   indices = [fill(Colon(), N-1)..., k]
   slice(A, indices...)
end

"Indexes for viewing into last axis of 4d array"
view(k::Integer) = (Colon(), Colon(), Colon(), k)
  
N = 32
Nh = N÷2 + 1

#Ui = rand(Float64, N, N, N)
#Wi = Array{Complex128}(Nh, N, N)
#
#
#const RFFT = plan_rfft(Ui, (1, 2, 3))
#
#fftn_mpi!(u, fu) = A_mul_B!(fu, RFFT, u)


dU = rand(Complex128, Nh, N, N, 3)
dU0 = deepcopy(dU)
dU1 = deepcopy(dU)
mask = rand(Bool, Nh, N, N)

dU[:] = dU.*mask
broadcast!(*, dU0, dU0, mask)

println(pointer(dU1))
dU1 .*= mask  # Not in place
println(pointer(dU1))

dU2 = broadcast(*, dU, mask)

(sumabs2(dU-dU0), sumabs2(dU-dU1), sumabs2(dU-dU2))
