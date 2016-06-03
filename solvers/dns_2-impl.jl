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

function dns(N)
    @assert N > 0 && (N & (N-1)) == 0 "N must be a power of 2"

    const comm = MPI.COMM_WORLD
    const rank = MPI.Comm_rank(comm)
    const num_processes = MPI.Comm_size(comm)
    @assert num_processes == 1 || num_processes % 2 == 0 "Need even number of workers"
    const nu = 0.000625
    const dt = 0.01
    const T = 0.1
    const Nh = N÷2+1
    const Np = N÷num_processes

    # Real vectors
    U = Array{Float64}(N, N, Np, 3)
    curl = similar(U)
    # Complex vectors
    dU = Array{Complex{Float64}}(Nh, Np, N, 3)
    U_hat, U_hat0, U_hat1 = similar(dU), similar(dU), similar(dU)
    # MPI 
    Uc_hat  = Array{Complex{Float64}}(Nh, Np, N)
    Uc_hatT = Array{Complex{Float64}}(Nh, N, Np)
    # Complex scalar
    P_hat = similar(Uc_hat)
    # Real grid
    x = collect(0:N-1)*2*pi/N
    X = similar(U)
    for (i, Xi) in enumerate(ndgrid(x, x, x[rank*Np+1:(rank+1)*Np])) X[view(i)...] = Xi end
    # Wave number grid
    kx = fftfreq(N, 1./N)
    kz = kx[1:(N÷2+1)]; kz[end] *= -1
    K = Array{Float64}(Nh, Np, N, 3)
    for (i, Ki) in enumerate(ndgrid(kz, kx[rank*Np+1:(rank+1)*Np], kx)) K[view(i)...] = Ki end
    # Square of wave number vectors
    K2 = reshape(sumabs2(K, 4), Nh, Np, N)
    # K/K2
    K_over_K2 = K./K2             
    # Fix division by zero
    if rank == 0
        for i in 1:3 K_over_K2[1, 1, 1, i] = K[1, 1, 1, i] end
    end
    # Dealising mask
    const kmax_dealias = 2*Nh/3
    dealias = reshape(reduce(&, [abs(K(i)) .< kmax_dealias for i in 1:3]), Nh, Np, N)
    # Runge-Kutta weights
    a = dt*[1./6., 1./3., 1./3., 1./6.]  
    b = dt*[0.5, 0.5, 1.] 
    # Work arrays for cross
    wcross = Array{eltype(U)}(N, N, Np)
    wcurl = Array{eltype(dU)}(Nh, Np, N)

    # Define FFT from plan
    const RFFT2 = plan_rfft(U(1), (1, 2))
    const FFTZ = plan_fft(Uc_hat, (3,))
    
    Uc_hatT_view = reshape(Uc_hatT, (Nh, Np, num_processes, Np))
    Uc_hat_view  = reshape(Uc_hat , (Nh, Np, Np, num_processes))
    
    Uc_hatr = similar(Uc_hat)
    Uc_hatr_view = reshape(Uc_hatr , (Nh, Np, Np, num_processes))
    
    "fftn from dns.py"
    function fftn_mpi!(u, fu)
      A_mul_B!(Uc_hatT, RFFT2, u)   # U c_hatT[:] = RFFT2*u
      permutedims!(Uc_hat_view, Uc_hatT_view, (1, 2, 4, 3))
      #Uc_hat_view[:] = MPI.Alltoall!(Uc_hat_view, Nh*Np*Np, comm)
      # A_mul_B!(fu, FFTZ, Uc_hat)    # fu[:] = FFTZ*Uc_hat
      MPI.Alltoall!(Uc_hatr_view, Uc_hat_view, Nh*Np*Np, comm)
      A_mul_B!(fu, FFTZ, Uc_hatr)    # fu[:] = FFTZ*Uc_hat  # hat_viewc, hatc
    end
    
    const IRFFT2 = plan_irfft(Uc_hatT, N, (1, 2))
    const IFFTZ = plan_ifft(Uc_hat, (3,))
    "ifftn from dns.py"
    function ifftn_mpi!(fu, u)
       A_mul_B!(Uc_hat, IFFTZ, fu)   # Uc_hat[:] = IFFTZ*fu
       # Uc_hat_view[:] = MPI.Alltoall!(Uc_hat_view, Nh*Np*Np, comm)
       # permutedims!(Uc_hatT_view, Uc_hat_view, (1, 2, 4, 3))
       MPI.Alltoall!(Uc_hatr_view, Uc_hat_view, Nh*Np*Np, comm)
       permutedims!(Uc_hatT_view, Uc_hatr_view, (1, 2, 4, 3))
       A_mul_B!(u, IRFFT2, Uc_hatT)  # u[:] = IRFFT2*Uc_hatT
    end

    function Cross!(w, a, b, c)
        for i in 1:3
            cross(i, a, b, w)
            fftn_mpi!(w, c(i))
        end
    end

    function Curl!(w, a, K, c, dealias)
        for i in 3:-1:1
            cross(i, K, a, w)
            scale!(w, im)
            broadcast!(*, w, w, dealias) 
            ifftn_mpi!(w, c(i))
        end
    end

    "sources, rk, out"
    function ComputeRHS!(wcross, wcurl, U, U_hat, curl, K, K_over_K2, K2, P_hat, nu, rk, dU)
        for i in 1:3 
            broadcast!(*, wcurl, U_hat[view(i)...], dealias)  # Use wcurl as work array
            ifftn_mpi!(wcurl, U(i))
        end

        Curl!(wcurl, U_hat, K, curl, dealias)
        Cross!(wcross, U, curl, dU)

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

    for i in 1:3 fftn_mpi!(U(i), U_hat(i)) end

    t = 0.0
    tstep = 0
    t_min, t_max = NaN, 0
    while t < T-1e-8
        tic()
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

        time_step = toq()
        t_min = min(time_step, t_min)
        t_max = max(time_step, t_max)
    end

    for i in 1:3 ifftn_mpi!(U_hat[view(i)...], U(i)) end
    
    k = MPI.Reduce(0.5*sumabs2(U)*(1./N)^3, MPI.SUM, 0, comm)
    t_min = MPI.Reduce(t_min, MPI.MIN, 0, comm)
    t_max = MPI.Reduce(t_max, MPI.MAX, 0, comm)
    if rank == 0
      println("$(k), $(t_min) $(t_max)")
    end
    (k, t_min, t_max)
end
