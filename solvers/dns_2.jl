import MPI


# Stuff that python imports.
"numpy.mgrid[v1, v2]"
function ndgrid{T}(v1::AbstractVector{T}, v2::AbstractVector{T})
    m, n = length(v1), length(v2)
    v1 = reshape(v1, m, 1)
    v2 = reshape(v2, 1, n)
    (repmat(v1, 1, n), repmat(v2, m, 1))
end

"helper"
function ndgrid_fill(a, v, s, snext)
    for j = 1:length(a)
        a[j] = v[div(rem(j-1, snext), s)+1]
    end
end

"numpy.mgrid[v1, v2, v3, ...]"
function ndgrid{T}(vs::AbstractVector{T}...)
    n = length(vs)
    sz = map(length, vs)
    out = ntuple(i->Array{T}(sz), n)
    s = 1
    for i=1:n
        a = out[i]::Array
        v = vs[i]
        snext = s*size(a,i)
        ndgrid_fill(a, v, s, snext)
        s = snext
    end
    out
end

"numpy.fft.fftfreq"
function fftfreq(n::Int, d::Real=1.0)
    val = 1.0/(n*d)
    results = Array{Int}(n)
    N = (n-1)÷2 + 1
    p1 = 0:(N-1)
    results[1:N] = p1
    p2 = -n÷2:-1
    results[N+1:end] = p2
    results * val
end

"View of A along the last axis"
function _{T, N}(A::AbstractArray{T, N}, k::Integer)
   # @assert 1 <= k <= last(size(A))
   indices = [fill(Colon(), N-1)..., k]
   slice(A, indices...)
end

"Indexes for viewing into last axis of 4d array"
view(k::Integer) = (Colon(), Colon(), Colon(), k)

"Linear indexing along last axis"
function linind{T, N}(A::AbstractArray{T, N})
    L = prod(size(A)[1:N-1])
    indices = [1]
    for k in 1:last(size(A)) push!(indices, last(indices)+L) end
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
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    num_processes = MPI.Comm_size(comm)

    const nu = 0.000625
    const dt = 0.01
    const T = 0.1
    const Nh = N÷2+1
    const Np = N÷num_processes

    # Real vectors
    U = Array{Float64}(N, N, Np, 3)
    curl = similar(U)
    # Complex vectors
    dU = Array{Complex128}(Nh, Np, N, 3)
    U_hat, U_hat0, U_hat1 = similar(dU), similar(dU), similar(dU)
    # MPI 
    Uc_hatT = Array{Complex128}(Nh, N, Np)
    Uc_hat  = Array{Complex128}(Nh, Np, N)
    
    # Complex scalars
    P_hat = Array{Complex128}(Nh, Np, N)
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
    dealias = reshape(reduce(&, [abs(_(K, i)) .< kmax_dealias for i in 1:3]), Nh, Np, N)
    # Runge-Kutta weights
    a = dt*[1./6., 1./3., 1./3., 1./6.]  
    b = dt*[0.5, 0.5, 1.] 
    # Work arrays for cross
    wcross = Array{eltype(U)}(N, N, Np)
    wcurl = Array{eltype(dU)}(Nh, Np, N)

    # Define FFT from plan
    const RFFT2 = plan_rfft(_(U, 1), (1, 2))
    const FFTZ = plan_fft(Uc_hat, (3,))
    
    Uc_hatT_view = reshape(Uc_hatT, (Nh, Np, num_processes, Np))
    Uc_hat_view  = reshape(Uc_hat , (Nh, Np, Np, num_processes))
    
    "fftn from dns.py"
    function fftn_mpi!(u, fu)
      Uc_hatT[:] = RFFT2*u
      permutedims!(Uc_hat_view, Uc_hatT_view, (1,2,4,3))
      Uc_hat_view[:,:,:,:] = MPI.Alltoall(Uc_hat_view, Nh*Np*Np, comm)
      fu[:] = FFTZ*Uc_hat
    end
    
    "ifftn from dns.py"
    const IRFFT2 = plan_irfft(Uc_hatT, N, (1, 2))
    const IFFTZ = plan_ifft(Uc_hat, (3,))
    function ifftn_mpi!(fu, u)
       Uc_hat[:] = IFFTZ*fu
       Uc_hat_view[:,:,:,:] = MPI.Alltoall(Uc_hat_view, Nh*Np*Np, comm)
       permutedims!(Uc_hatT_view, Uc_hat_view, (1,2,4,3))
       u[:] = IRFFT2*Uc_hatT
    end

    function Cross!(w, a, b, c)
        for i in 1:3
            cross(i, a, b, w)
            fftn_mpi!(w, _(c, i))
        end
    end

    function Curl!(w, a, K, c, dealias)
        for i in 3:-1:1
            cross(i, K, a, w)
            scale!(w, im)
            broadcast!(*, w, w, dealias) 
            ifftn_mpi!(w, _(c, i))
        end
    end

    "sources, rk, out"
    function ComputeRHS!(wcross, wcurl, U, U_hat, curl, K, K_over_K2, K2, P_hat, nu, rk, dU)
        for i in 1:3 ifftn_mpi!(U_hat[view(i)...].*dealias, _(U, i)) end

        Curl!(wcurl, U_hat, K, curl, dealias)
        
        Cross!(wcross, U, curl, dU)
#         broadcast!(*, dU, dU, dealias)

        # P_hat[:] = sum(dU.*K_over_K2, 4)
        P_hat[:] = 0im
        indices = linind(dU)
        for axis in 1:last(size(dU))
            for (j, i) in enumerate(indices[axis]:indices[axis+1]-1)
                @inbounds P_hat[j] += dU[i]*K_over_K2[i]
            end
        end

        #axpy!(-1., broadcast(*, P_hat, K), dU)
        #axpy!(-1., nu*broadcast(*, U_hat, K2), dU)
        for axis in 1:last(size(dU))
            for (j, i) in enumerate(indices[axis]:indices[axis+1]-1)
                @inbounds dU[i] -= P_hat[j]*K[i] + nu*U_hat[i]*K2[j]
            end
        end

    end

    U[view(1)...] = sin(_(X, 1)).*cos(_(X, 2)).*cos(_(X, 3))
    U[view(2)...] = -cos(_(X, 1)).*sin(_(X, 2)).*cos(_(X, 3))
    U[view(3)...] = 0.

    for i in 1:3 fftn_mpi!(_(U, i), _(U_hat, i)) end

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
        for i in 1:3 ifftn_mpi!(U_hat[view(i)...], _(U, i)) end
    end
    one_step = toq()/tstep

    for i in 1:3 ifftn_mpi!(U_hat[view(i)...], _(U, i)) end
    
    k = MPI.Reduce(0.5*sumabs2(U)*(1./N)^3, MPI.SUM, 0, comm)
    if rank == 0
      println("$(k), $(one_step)")
    end
    
    MPI.Finalize()
    
end

dns(2^7)

