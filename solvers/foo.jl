# Generated stuff
using Base.Cartesian

@generated function cross{T1, T2, T3}(k::Int,
                                      X::AbstractArray{T1, 4},
                                      Y::AbstractArray{T2, 4},
                                      w::AbstractArray{T3, 3})
    quote
        @assert size(X) == size(Y) && size(X)[1:3] == size(w)
        kp, kpp = (k+1-1)%3+1, (k+2-1)%3+1
        @nloops 3 i w begin
            @inbounds (@nref 3 w i) = 
            (@nref 4 X d->(d<4)?i_d:kp)*(@nref 4 Y d->(d<4)?i_d:kpp)-
            (@nref 4 X d->(d<4)?i_d:kpp)*(@nref 4 Y d->(d<4)?i_d:kp)
        end
    end
end

@generated function cross{T1, T2, T3}(X::AbstractArray{T1, 4},
                                      Y::AbstractArray{T2, 4},
                                      W::AbstractArray{T3, 4})
    quote
        @assert size(X) == size(Y) && size(X) == size(W)
        @nloops 4 i W d->(d == 4)?(kp = (i_4+1-1)%3+1; kpp = (i_4+2-1)%3+1):nothing begin
            @inbounds (@nref 4 W i) = 
            (@nref 4 X d->(d<4)?i_d:kp)*(@nref 4 Y d->(d<4)?i_d:kpp)-
            (@nref 4 X d->(d<4)?i_d:kpp)*(@nref 4 Y d->(d<4)?i_d:kp)
        end
    end
end

N = 64
X = rand(N, N, N, 3)
Y = rand(N, N, N, 3)
w = rand(N, N, N)

W = similar(X)
cross(X, Y, W)

for i in 1:3
    cross(i, X, Y, w)
    println("$(sumabs2(w-W[:, :, :, i]))")
end
