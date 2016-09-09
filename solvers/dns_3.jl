import MPI

import mpiFFT4jl.slab: r2c, rfft3, irfft3, real_shape, complex_shape,
                       get_local_mesh, get_local_wavenumbermesh

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

# ----------------------------------------------------------------------------

using Base.LinAlg.BLAS: axpy!

function dns(n)
    @assert n > 0 && (n & (n-1)) == 0 "N must be a power of 2"

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    num_processes = MPI.Comm_size(comm)
    
#     FFTW.set_num_threads(2)

    const nu = 0.000625
    const dt = 0.01
    const T = 0.1
    const N = [n, n, n]    # Global shape of mesh
    const L = [2pi, 2pi, 2pi] # Real size of mesh
    
    FFT = r2c(N, L, comm)

    rshape = real_shape(FFT)
    rvector_shape = tuple(push!([rshape...], 3)...)    
    cshape = complex_shape(FFT)
    cvector_shape = tuple(push!([cshape...], 3)...)    
    # Real vectors
    U = Array{Float64}(rvector_shape)
    curl = similar(U)
    # Complex vectors
    dU = Array{Complex{Float64}}(cvector_shape)
    U_hat, U_hat0, U_hat1 = similar(dU), similar(dU), similar(dU)
    # Complex scalar
    P_hat = zeros(Complex{Float64}, cshape)
    # Real grid
    X = get_local_mesh(FFT)
    
    # Wave number grid
    K = get_local_wavenumbermesh(FFT)
    
    # Square of wave number vectors
    K2 = reshape(sumabs2(K, 4), cshape)
    
    # K/K2
    K_over_K2 = K./K2             
    # Fix division by zero
    if rank == 0
        for i in 1:3 K_over_K2[1, 1, 1, i] = K[1, 1, 1, i] end
    end
    
    # Runge-Kutta weights
    a = dt*[1./6., 1./3., 1./3., 1./6.]  
    b = dt*[0.5, 0.5, 1.] 
    # Work arrays for cross
    wcross = Array{eltype(U)}(rshape)
    wcurl = Array{eltype(dU)}(cshape)

    function Cross!(w, a, b, c, FFT)
        for i in 1:3
            cross(i, a, b, w)
            rfft3(FFT, c(i), w)
        end
    end

    function Curl!(w, a, K, c, FFT)
        for i in 3:-1:1
            cross(i, K, a, w)
            scale!(w, im)
            irfft3(FFT, c(i), w, 1)
        end
    end

    "sources, rk, out"
    function ComputeRHS!(wcross, wcurl, U, U_hat, curl, K, K_over_K2, K2, P_hat, nu, rk, dU, FFT)
        for i in 1:3 irfft3(FFT, U(i), U_hat[view(i)...], 1) end

        Curl!(wcurl, U_hat, K, curl, FFT)
        Cross!(wcross, U, curl, dU, FFT)

        P_hat[:] = zero(eltype(P_hat))
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

    for i in 1:3 rfft3(FFT,  U_hat(i), U(i)) end
        
    t = 0.0
    tstep = 0
    t_min, t_max = NaN, 0
    while t < T-1e-8
        tic()

        t += dt; tstep += 1
        U_hat1[:] = U_hat; U_hat0[:] = U_hat
        
        for rk in 1:4
            ComputeRHS!(wcross, wcurl, U, U_hat, curl, K, K_over_K2, K2, P_hat, nu, rk, dU, FFT)
            if rk < 4
                U_hat[:] = U_hat0
                axpy!(b[rk], dU, U_hat)
            end
            axpy!(a[rk], dU, U_hat1)
        end
        U_hat[:] = U_hat1
        for i in 1:3 irfft3(FFT, U(i), U_hat[view(i)...]) end

        time_step = toq()
        t_min = min(time_step, t_min)
        t_max = max(time_step, t_max)
    end
    
    k = MPI.Reduce(0.5*sumabs2(U)*(1./prod(FFT.N)), MPI.SUM, 0, comm)
    t_min = MPI.Reduce(t_min, MPI.MIN, 0, comm)
    t_max = MPI.Reduce(t_max, MPI.MAX, 0, comm)
    (k, t_min)
end

