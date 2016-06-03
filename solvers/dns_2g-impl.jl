include("utils.jl")
using Utils  # Now fftfreq and ndgrid are available
using Base.Cartesian

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

"Component of the cross product [X \times Y]_k = w"
function cross!{T1, T2, T3}(k::Int,
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
    Uc_hat = Array{Complex{Float64}}(Nh, Np, N)
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
            cross!(i, a, b, w)
            fftn_mpi!(w, c(i))
        end
    end

    function Curl!(w, a, K, c, dealias)
        for i in 3:-1:1
            cross!(i, K, a, w)
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
        @nloops 4 i dU begin
            @inbounds (@nref 3 P_hat i) += (@nref 4 dU i) * (@nref 4 K_over_K2 i)
        end

        @nloops 4 i dU begin
            @inbounds (@nref 4 dU i) -= (@nref 3 P_hat i)*(@nref 4 K i) + nu*(@nref 4 U_hat i)*(@nref 3 K2 i)
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

    for i in 1:3 ifftn_mpi!(U_hat[view(i)...], U(i)) end
    
    k = MPI.Reduce(0.5*sumabs2(U)*(1./N)^3, MPI.SUM, 0, comm)
    if rank == 0
      println("$(k), $(one_step)")
    end
    (k, one_step)
end
