# Speed and corretness of various implmentations of the cross product

"Linear indexing along last axis"
function linind{T, N}(A::AbstractArray{T, N})
    L = prod(size(A)[1:N-1])
    indices = [1]
    for k in 1:last(size(A)) push!(indices, last(indices)+L) end
    indices
end

function Cross(X, Y, W)
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

function cross{Xtype, Ytype, Wtype}(kaxis::Int,
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

# Generated stuff
using Base.Cartesian

function crossG{T1, T2, T3}(k::Int,
                            X::AbstractArray{T1, 4},
                            Y::AbstractArray{T2, 4},
                            w::AbstractArray{T3, 3})
    @assert size(X) == size(Y) && size(X)[1:3] == size(w)
    kp, kpp = (k+1-1)%3+1, (k+2-1)%3+1
    @nloops 3 i w begin
        @inbounds (@nref 3 w i) = 
        (@nref 4 X d->(d<4)?i_d:kp)*(@nref 4 Y d->(d<4)?i_d:kpp)-
        (@nref 4 X d->(d<4)?i_d:kpp)*(@nref 4 Y d->(d<4)?i_d:kp)
    end
end

function CrossG{T1, T2, T3}(X::AbstractArray{T1, 4},
                            Y::AbstractArray{T2, 4},
                            W::AbstractArray{T3, 4})
    @assert size(X) == size(Y) && size(X) == size(W)
    @nloops 4 i W d->(d == 4)?(kp = (i_4+1-1)%3+1; kpp = (i_4+2-1)%3+1):nothing begin
        @inbounds (@nref 4 W i) = 
        (@nref 4 X d -> if d<4 i_d else kp end)*(@nref 4 Y d -> if d < 4  i_d else kpp end)-
        (@nref 4 X d -> if d<4 i_d else kpp end)*(@nref 4 Y d -> if d < 4  i_d else kp end)
        # X[i_1, i_2, i_3, kp]*Y[i_1, i_2, i_3, kpp]-X[i_1, i_2, i_3, kpp]*Y[i_1, i_2, i_3, kp]
    end
end

function CrossH{T1, T2, T3}(X::AbstractArray{T1, 4},
                            Y::AbstractArray{T2, 4},
                            W::AbstractArray{T3, 4})
    @assert size(X) == size(Y) && size(X) == size(W)
    for i4 in 1:size(W, 4)
        kp = (i4+1-1)%3+1
        kpp = (i4+2-1)%3+1
        for i3 in 1:size(W, 3)
            for i2 in 1:size(W, 2)
                for i1 in 1:size(W, 1)
                    @inbounds W[i1, i2, i3, i4] = X[i1, i2, i3, kp]*Y[i1, i2, i3, kpp] - X[i1, i2, i3, kpp]*Y[i1, i2, i3, kp]
                end
            end
        end
    end
end

function crossH{T1, T2, T3}(k::Int,
                            X::AbstractArray{T1, 4},
                            Y::AbstractArray{T2, 4},
                            w::AbstractArray{T3, 3})
    @assert size(X) == size(Y) && size(X)[1:3] == size(w)
    kp = (k+1-1)%3+1
    kpp = (k+2-1)%3+1

    for i3 in 1:size(w, 3)
        for i2 in 1:size(w, 2)
            for i1 in 1:size(w, 1)
                @inbounds w[i1, i2, i3] = X[i1, i2, i3, kp]*Y[i1, i2, i3, kpp] - X[i1, i2, i3, kpp]*Y[i1, i2, i3, kp]
            end
        end
    end
end

# ----------------------------------------------------------------------------

n = 2^6
X = rand(n, n, n, 3)
Y = rand(n, n, n, 3)
W = rand(n, n, n, 3)
w = rand(n, n, n)

Cross(X, Y, W)
CrossG(X, Y, W)
CrossH(X, Y, W)
# Speed 
@time for i in 1:100 Cross(X, Y, W) end
@time for i in 1:100 CrossG(X, Y, W) end
@time for i in 1:100 CrossH(X, Y, W) end
# Correctness
WG = similar(W); CrossG(X, Y, WG)
WH = similar(W);
CrossH(X, Y, WH)
Cross(X, Y, W)
println(sumabs2(W-WG))
println(sumabs2(W-WH))


cross(1, X, Y, w)
crossG(1, X, Y, w)
crossH(1, X, Y, w)
# Speed
@time for i in 1:5 cross(1, X, Y, w) end
@time for i in 1:5 crossG(1, X, Y, w) end
@time for i in 1:5 crossH(1, X, Y, w) end
# Correctness
cross(2, X, Y, w)
wG = similar(w); crossG(2, X, Y, wG)
wH = similar(w); crossH(2, X, Y, wH)
println(sumabs2(w-wG))
println(sumabs2(w-wH))
