include("utils.jl")
using Utils  # Now fftfreq and ndgrid are available

# NOTE: Let A = rand(3, 3)
# 1. A[:, 1] = rand(3)               Assigns to first column of A
# 2. slice(A, :, 1) = rand(3)        Assigns to first column of A
# 3. copy!(A[:, 1], rand(3))         Leaves A alone
# 4. copy!(slice(A, :, 1), rand(3))  Assigns to first column of A
# Mainly to use 1. and 3. we want a shortcut for fixing last axis
view(k::Int, N::Int=4) = [fill(Colon(), N-1)..., k]

"View of A with last coordinate fixed at k"
function call{T, N}(A::AbstractArray{T, N}, k::Int)
   @assert 1 <= k <= size(A, N)
   indices = [fill(Colon(), N-1)..., k]
   slice(A, indices...)
end

"Linear indexing along last axis"
function linind{T, N}(A::AbstractArray{T, N})
    L = prod(size(A)[1:N-1])
    indices = [1]
    for k in 1:size(A, N) push!(indices, last(indices)+L) end
    indices
end

"Component of the cross product [X \times Y]_k = w"
function cross!{S, T, R}(kaxis::Int, X::AbstractArray{S, 4},
                                     Y::AbstractArray{T, 4},
                                     w::AbstractArray{R, 3})
    @assert 1 <= kaxis <= 3 && size(X) == size(Y) && size(X)[1:3] == size(w)

    indices = linind(X)
    axis = [1, 2, 3, 1, 2]
    iaxis, jaxis = axis[kaxis+1], axis[kaxis+2]
    iindexes = indices[iaxis]:indices[iaxis+1]-1
    jindexes = indices[jaxis]:indices[jaxis+1]-1
    for (k, (i, j)) in enumerate(zip(iindexes, jindexes))
        @inbounds w[k] = X[i]*Y[j] - X[j]*Y[i]
    end
end

# ----------------------------------------------------------------------------

using Base.LinAlg.BLAS: axpy!

function dns(N)
    const nu = 0.000625
    const dt = 0.01
    const T = 0.1
    const Nh = N÷2+1

    typealias MyReal Float64
    # Real vectors
    U = Array{MyReal}(N, N, N, 3)
    curl = similar(U)
    # Complex vectors
    dU = Array{Complex{MyReal}}(Nh, N, N, 3)
    U_hat, U_hat0, U_hat1 = similar(dU), similar(dU), similar(dU)
    # Complex scalars
    P_hat = Array{eltype(dU)}(Nh, N, N)
    # Real grid
    x = collect(0:N-1)*2*pi/N
    X = similar(U)
    for (i, Xi) in enumerate(ndgrid(x, x, x)) X[view(i)...] = Xi end
    # Wave number grid
    kx = fftfreq(N, 1./N)
    kz = kx[1:(N÷2+1)]; kz[end] *= -1
    K = Array{eltype(U)}(Nh, N, N, 3)
    for (i, Ki) in enumerate(ndgrid(kz, kx, kx)) K[view(i)...] = Ki end
    # Square of wave number vectors
    K2 = Array{eltype(U)}(Nh, N, N)
    sumabs2!(K2, K)
    # K/K2
    K_over_K2 = K./K2             
    # Fix division by zero
    for i in 1:3 K_over_K2[1, 1, 1, i] = K[1, 1, 1, i] end
    # Dealising mask
    const kmax_dealias = 2*Nh/3
    dealias = reshape(reduce(&, [abs(K(i)) .< kmax_dealias for i in 1:3]), Nh, N, N)
    # Runge-Kutta weights
    a = dt*[1./6., 1./3., 1./3., 1./6.]  
    b = dt*[0.5, 0.5, 1.] 
    # Work arrays for cross
    wcross = Array{eltype(U)}(N, N, N)
    wcurl = Array{eltype(dU)}(Nh, N, N)

    # Define (I)RFFTs
    const RFFT = plan_rfft(wcross, (1, 2, 3))
    fftn_mpi!(u, fu) = A_mul_B!(fu, RFFT, u)

    const IRFFT = plan_irfft(wcurl, N, (1, 2, 3))
    ifftn_mpi!(fu, u) = A_mul_B!(u, IRFFT, fu)

    function Cross!(w, a, b, c)
        for i in 1:3
            cross!(i, a, b, w)
            fftn_mpi!(w, c(i))
        end
    end

    function Curl!(w, a, K, c)
        for i in 3:-1:1
            cross!(i, K, a, w)
            scale!(w, im)
            ifftn_mpi!(w, c(i))
        end
    end

    function ComputeRHS!(wcross, wcurl, U, U_hat, curl, K, K_over_K2, K2, P_hat, nu, rk, dU)
        if rk > 1
            for i in 1:3 ifftn_mpi!(U_hat[view(i)...], U(i)) end
        end

        Curl!(wcurl, U_hat, K, curl)
        Cross!(wcross, U, curl, dU)
        broadcast!(*, dU, dU, dealias)

        P_hat[:] = 0im
        indices = linind(dU)
        for axis in 1:last(size(dU))
            for (j, i) in enumerate(indices[axis]:indices[axis+1]-1)
                @inbounds P_hat[j] += dU[i]*K_over_K2[i]
            end
        end

        for axis in 1:last(size(dU))
            for (j, i) in enumerate(indices[axis]:indices[axis+1]-1)
                @inbounds dU[i] -= P_hat[j]*K[i] + nu*U_hat[i]*K2[j]
            end
        end
    end

    U[view(1)...] = sin(X(1)).*cos(X(2)).*cos(X(3))
    U[view(2)...] = -cos(X(1)).*sin(X(2)).*cos(X(3))
    U[view(3)...] = 0.

    for i in 1:3 fftn_mpi!(U(i), U_hat(i)) end

    t = 0.0
    tstep = 0
    tic()
    while t < T-1e-8
        t += dt; tstep += 1
        U_hat1[:] = U_hat; U_hat0[:] = U_hat
        
        for rk in 1:4
            ComputeRHS!(wcross, wcurl, U, U_hat, curl, K, K_over_K2, K2, P_hat, nu, rk, dU)
            if rk < 4
                U_hat[:] = U_hat0
                axpy!(b[rk], dU, U_hat)
            end
            axpy!(a[rk], dU, U_hat1)
        end

        U_hat[:] = U_hat1
        for i in 1:3 ifftn_mpi!(U_hat[view(i)...], U(i)) end
    end
    one_step = toq()/tstep

    k = 0.5*sumabs2(U)*(1./N)^3
    (k, one_step)
end
