# dns_2g with code object based FFTs
include("utils.jl")
using Utils  # Now fftfreq and ndgrid are available
using Base.Cartesian

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

# Spectral transformation of three dimensional data aligned
# such that the last component is parallelized across processes
immutable SpecTransf{T<:Real}
    # Plans
    plan12::FFTW.rFFTWPlan{T}
    plan3::FFTW.cFFTWPlan
    # Work arrays for transformations
    vT::Array{Complex{T}, 3}
    vT_view::Array{Complex{T}, 4}
    v::Array{Complex{T}, 3}
    v_view::Array{Complex{T}, 4}
    v_recv::Array{Complex{T}, 3}
    v_recv_view::Array{Complex{T}, 4}
    # Communicator
    comm::MPI.Comm
    # Amount of data to be send by MPI
    chunk::Int

    # Inner constructor
    function SpecTransf(A, comm)
        # Verify input
        sizes = size(A)
        N = first(sizes)
        Nh = N÷2+1
        p = MPI.Comm_size(comm)
        Np = N÷p
        @assert size(A) == (N, N, Np)

        # Allocate work arrays
        vT, v = Array{Complex{T}}(Nh, N, Np), Array{Complex{T}}(Nh, Np, N)
        vT_view, v_view = reshape(vT, (Nh, Np, p, Np)), reshape(v, (Nh, Np, Np, p))
        # For MPI.Alltoall! preallocate the receiving buffer
        v_recv = similar(v); v_recv_view = reshape(v_recv, (Nh, Np, Np, p))

        # Plan Fourier transformations
        plan12 = plan_rfft(A, (1, 2))
        plan3 = plan_fft!(v, (3, ))
        # Compute the inverse plans
        inv(plan12); inv(plan3)

        chunk = Nh*Np*Np
        # Now we are ready
        new(plan12, plan3,
            vT, vT_view, v, v_view, v_recv, v_recv_view,
            comm, chunk)
    end
end

# Constructor
SpecTransf{T<:Real}(A::AbstractArray{T, 3}, comm::Any) = SpecTransf{T}(A, comm)

# Transform real to complex as complex = T o real
function apply{T<:Real}(fu::AbstractArray{Complex{T}, 3}, F::SpecTransf{T}, u::AbstractArray{T})
    A_mul_B!(F.vT, F.plan12, u)
    permutedims!(F.v_view, F.vT_view, [1, 2, 4, 3])
    MPI.Alltoall!(F.v_recv_view, F.v_view, F.chunk, F.comm)
    F.plan3*F.v_recv; fu[:] = F.v_recv
end

# Transform complex to real as real = T o complex
function apply_inv{T<:Real}(u::AbstractArray{T}, F::SpecTransf{T}, fu::AbstractArray{Complex{T}, 3})
    F.plan3.pinv*fu; F.v[:] = fu
    MPI.Alltoall!(F.v_recv_view, F.v_view, F.chunk, F.comm)
    permutedims!(F.vT_view, F.v_recv_view, [1, 2, 4, 3])
    A_mul_B!(u, F.plan12.pinv, F.vT)
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

    # Instantiate spectral transformation
    const F = SpecTransf(wcross, comm) 

    function Cross!(F, w, a, b, c)
        for i in 1:3
            cross!(i, a, b, w)
            apply(c(i), F, w)
        end
    end

    function Curl!(F, w, a, K, c, dealias)
        for i in 3:-1:1
            cross!(i, K, a, w)
            scale!(w, im)
            broadcast!(*, w, w, dealias) 
            apply_inv(c(i), F, w)
        end
    end

    "sources, rk, out"
    function ComputeRHS!(F, wcross, wcurl, 
                         U, U_hat, curl, K, K_over_K2, K2, P_hat, nu, rk, dU)
        for i in 1:3 
            broadcast!(*, wcurl, U_hat[view(i)...], dealias)  # Use wcurl as work array
            apply_inv(U(i), F, wcurl)
        end

        Curl!(F, wcurl, U_hat, K, curl, dealias)
        Cross!(F, wcross, U, curl, dU)

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

    for i in 1:3 apply(U_hat(i), F, U(i)) end

    t = 0.0
    tstep = 0
    t_min, t_max = NaN, 0
    while t < T-1e-8
        tic()

        t += dt; tstep += 1
        U_hat1[:] = U_hat; U_hat0[:] = U_hat
        
        for rk in 1:4
            ComputeRHS!(F, wcross, wcurl,
                        U, U_hat, curl, K, K_over_K2, K2, P_hat, nu, rk, dU)
            if rk < 4
                U_hat[:] = U_hat0
                axpy!(b[rk], dU, U_hat)
            end
            axpy!(a[rk], dU, U_hat1)
        end
        U_hat[:] = U_hat1
        for i in 1:3 apply_inv(U(i), F, U_hat[view(i)...]) end

        time_step = toq()
        t_min = min(time_step, t_min)
        t_max = max(time_step, t_max)
    end

    for i in 1:3 apply_inv(U(i), F, U_hat[view(i)...]) end
    
    k = MPI.Reduce(0.5*sumabs2(U)*(1./N)^3, MPI.SUM, 0, comm)
    t_min = MPI.Reduce(t_min, MPI.MIN, 0, comm)
    t_max = MPI.Reduce(t_max, MPI.MAX, 0, comm)
    if rank == 0
      println("$(k), $(t_min) $(t_max)")
    end
    (k, t_min, t_max)
end
