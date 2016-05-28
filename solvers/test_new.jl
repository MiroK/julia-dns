"fftn from dns.py"
fftn_mpi!(u, fu) = fu[:] = rfft(u, (1, 2, 3))

"ifftn from dns.py"
ifftn_mpi!(fu, u) = u[:] = irfft(fu, first(size(u)), (1, 2, 3))

"View of A along the last axis"
function _{T, N}(A::AbstractArray{T, N}, k::Integer)
   @assert 1 <= k <= last(size(A))
   indices = [fill(Colon(), N-1)..., k]
   slice(A, indices...)
end

"Indexes for viewing into last axis of 4d array"
view(k::Integer) = (Colon(), Colon(), Colon(), k)

#-----------------------------------------------------------------------------

X = rand(16, 16, 16)
Y = rand(16, 16, 16)
Z = rand(16, 16, 16)

fX = Array{Complex{Float64}}(9, 16, 16)
fY = similar(fX)
fZ = similar(fX)

A = Array[X, Y, Z]
fA = Array[fX, fY, fZ]
ffA = deepcopy(A)

for i in 1:3 fftn_mpi!(A[i], fA[i]) end
for i in 1:3 ifftn_mpi!(fA[i], ffA[i]) end

for i in 1:3 println(maximum(A[i]-ffA[i])) end

# Now let's oragnize the data into one array
B = Array{Float64}(16, 16, 16, 3)
B[view(1)...] = X
B[view(2)...] = Y
B[view(3)...] = Z

fB = Array{Complex{Float64}}(9, 16, 16, 3) 
for i in 1:3 fftn_mpi!(_(B, i), _(fB, i)) end

# is fB same as fA
for i in 1:3 println(maximum(abs(fA[i] - _(fB, i)))) end

# Go back
ffB = similar(B)
for i in 1:3 ifftn_mpi!(_(fB, i), _(ffB, i)) end

for i in 1:3 println(maximum(abs(A[i] - _(ffB, i)))) end
