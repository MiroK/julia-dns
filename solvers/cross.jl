# Speed and corretness of various implmentations of the cross product

"View of A along the last axis"
function _{T, N}(A::AbstractArray{T, N}, k::Integer)
   @assert 1 <= k <= last(size(A))
   indices = [fill(Colon(), N-1)..., k]
   slice(A, indices...)
end

n = 2^6

X = rand(n, n, n, 3)
Y = rand(n, n, n, 3)
W1 = similar(X)
W2 = similar(X)
W3 = similar(X)
W4 = similar(X)

function cross_1(X, Y, W)
    _(W, 1)[:] = _(X, 2).*_(Y, 3)-_(X, 3).*_(Y, 2)
    _(W, 2)[:] = _(X, 3).*_(Y, 1)-_(X, 1).*_(Y, 3)
    _(W, 3)[:] = _(X, 1).*_(Y, 2)-_(X, 2).*_(Y, 1)
end

"Linear indexing along last axis"
function linind{T, N}(A::AbstractArray{T, N})
    L = prod(size(A)[1:N-1])
    indices = [1]
    for k in 1:last(size(A)) push!(indices, last(indices)+L) end
    indices
end

function cross_2(X, Y, W)
    axis = [1, 2, 3, 1, 2]
    indices = linind(W)
    for (kaxis, iaxis, jaxis) in zip(axis[1:3], axis[2:4], axis[3:5])
        kindexes = indices[kaxis]:indices[kaxis+1]-1
        iindexes = indices[iaxis]:indices[iaxis+1]-1
        jindexes = indices[jaxis]:indices[jaxis+1]-1
        for (k, i, j) in zip(kindexes, iindexes, jindexes)
            @inbounds W[k] = X[i]*Y[j] - X[j]*Y[i]
        end
    end
end

function cross{Xtype, Ytype, Wtype}(kaxis::Integer,
                                    X::AbstractArray{Xtype, 4},
                                    Y::AbstractArray{Ytype, 4},
                                    w::AbstractArray{Wtype, 3})
    @assert 1 <= kaxis <= 3
    @assert size(X) == size(Y)
    @assert size(X)[1:3] == size(w)

    indices = linind(X)
    axis = [1, 2, 3, 1, 2]
    iaxis, jaxis = axis[kaxis+1], axis[kaxis+2]
    iindexes = indices[iaxis]:indices[iaxis+1]-1
    jindexes = indices[jaxis]:indices[jaxis+1]-1
    for (k, (i, j)) in enumerate(zip(iindexes, jindexes))
        @inbounds w[k] = X[i]*Y[j] - X[j]*Y[i]
    end
end

function cross_3(X, Y, W)
    axis = [1, 2, 3, 1, 2]
    indices = linind(W)
    for k in 1:3
        Xi, Yj = _(X, axis[k+1]), _(Y, axis[k+2])
        Xj, Yi = _(X, axis[k+2]), _(Y, axis[k+1])
        @inbounds for (l, r) in zip(indices[k]:indices[k+1]-1, eachindex(Xi))
            W[l] = Xi[r]*Yj[r] - Xj[r]*Yi[r]
        end
    end
end

function cross_4(X, Y, W)
    axis = [1, 2, 3, 1, 2]
    indices = linind(W)
    for k in 1:3
        Xi, Yj = _(X, axis[k+1]), _(Y, axis[k+2])
        Xj, Yi = _(X, axis[k+2]), _(Y, axis[k+1])
        @inbounds for (l, xi, yj, xj, yi) in zip(indices[k]:indices[k+1]-1, Xi, Yj, Xj, Yi)
            W[l] = xi*yj - xj*yi
        end
    end
end

cross_1(X, Y, W1)
cross_2(X, Y, W2)
cross_3(X, Y, W3)
cross_4(X, Y, W4)
println(sumabs2(W1-W2))
println(sumabs2(W1-W3))
println(sumabs2(W1-W4))

w = rand(n, n, n) 
for i in 1:3
    cross(i, X, Y, w)
    println("$(i) $(sumabs2(w - _(W1, i)))")
end
