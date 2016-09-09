#! /usr/bin/julia
#=
Created on Fri 9 Sep 14:08:38 2016

@author: Diako Darian

3D periodic MHD-solver using Fourier-Galerkin method
=#

import MPI
using Iterators

"numpy.mgrid[v1, v2]"
function ndgrid{T}(v1::AbstractVector{T}, v2::AbstractVector{T})
    m, n = length(v1), length(v2)
    v1 = reshape(v1, m, 1)
    v2 = reshape(v2, 1, n)
    (repmat(v1, 1, n), repmat(v2, m, 1))
end

"Helper"
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

"numpy.fft.rfftfreq"
function rfftfreq(n::Int, d::Real=1.0)
    val = 1.0/(n*d)
    N = (n)÷2 + 1
    results = Array{Int}(N)
    results[1:N] = 0:(N-1)
    results * val
end

"Modifies while loop such that ivar and avar returns mpi-reduced minimal and
maximal times that it takes to execute once the body of the loop."
macro mpitime(loop, ivar, avar)
    _mpi_time(loop, ivar, avar)
end

function _mpi_time(loop, ivar, avar)
    loop.head != :while && error("Not a while loop")

    tstep = symbol(string(ivar), string(avar))

    # Modify internals of the loop
    internals = loop.args[2]
    internals.head == :quote && error("I did not expect this")
    insert!(internals.args, 1, :(tic()))          # Time the internals
    push!(internals.args, :($tstep = toq()))
    push!(internals.args, :($ivar = min($tstep, $ivar)))  # Update min/max
    push!(internals.args, :($avar = max($tstep, $avar)))

    body = quote
        $ivar, t_max = NaN, 0
        $loop
        $ivar = MPI.Reduce($ivar, MPI.MIN, 0, MPI.COMM_WORLD)
        $avar = MPI.Reduce($avar, MPI.MAX, 0, MPI.COMM_WORLD)
    end

    ex = Expr(:escape, body)
    ex
end

# Spectral transformation of three dimensional data aligned
# such that the last component is parallelized across processes
type r2c{T<:Real}
    # Global shape
    N::Array{Int, 1}
    # Global size of domain
    L::Array{T, 1}
    # Communicator
    comm::MPI.Comm
    num_processes::Int
    rank::Int
    chunk::Int    # Amount of data to be send by MPI

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
    dealias::Array{Int, 1}

    # Constructor
    function r2c(N, L, comm)
        # Verify input
        Nh = N[1]÷2+1
        p = MPI.Comm_size(comm)
        Np = N÷p

        # Allocate work arrays
        vT, v = Array{Complex{T}}(Nh, N[2], Np[3]), Array{Complex{T}}(Nh, Np[2], N[3])
        vT_view, v_view = reshape(vT, (Nh, Np[2], p, Np[3])), reshape(v, (Nh, Np[2], Np[3], p))
        # For MPI.Alltoall! preallocate the receiving buffer
        v_recv = similar(v); v_recv_view = reshape(v_recv, (Nh, Np[2], Np[3], p))

        # Plan Fourier transformations
        A = zeros(T, (N[1], N[2], Np[3]))
        if p > 1
            plan12 = plan_rfft(A, (1, 2))
            plan3 = plan_fft!(v, (3, ))
        else  # Use only plan12 to do entire transform
            plan12 = plan_rfft(A, (1, 2, 3))
            plan3 = plan_fft!(zeros(Complex{Float64}, 2,2,2), (1, 2, 3))
        end

        # Compute the inverse plans
        inv(plan12)
        if p > 1 inv(plan3) end

        chunk = Nh*Np[2]*Np[3]
        # Now we are ready
        new(N, L, comm, p, MPI.Comm_rank(comm), chunk,
            plan12, plan3,
            vT, vT_view, v, v_view, v_recv, v_recv_view)
    end
end

# Constructor
r2c{T<:Real}(N::Array{Int, 1}, L::Array{T, 1}, comm::Any) = r2c{T}(N, L, comm)

# Transform real to complex as complex = T o real
function rfft3{T<:Real}(F::r2c{T}, fu::AbstractArray{Complex{T}, 3}, u::AbstractArray{T})
    if F.num_processes > 1
        A_mul_B!(F.vT, F.plan12, u)
        permutedims!(F.v_view, F.vT_view, [1, 2, 4, 3])
        MPI.Alltoall!(F.v_recv_view, F.v_view, F.chunk, F.comm)
        F.plan3*F.v_recv; fu[:] = F.v_recv
    else
        A_mul_B!(fu, F.plan12, u)
    end
end

# Transform complex to real as real = T o complex
function irfft3{T<:Real}(F::r2c{T}, u::AbstractArray{T}, fu::AbstractArray{Complex{T}, 3}, dealias_fu::Int=0)
    if F.num_processes > 1
        F.v[:] = fu
        if dealias_fu == 1
            dealias(F, F.v)
        elseif dealias_fu == 2
            dealias2(F, F.v)
        end
        F.plan3.pinv*F.v
        MPI.Alltoall!(F.v_recv_view, F.v_view, F.chunk, F.comm)
        permutedims!(F.vT_view, F.v_recv_view, [1, 2, 4, 3])
        A_mul_B!(u, F.plan12.pinv, F.vT)
    else
        A_mul_B!(u, F.plan12.pinv, fu)
    end
end

function real_shape{T<:Real}(F::r2c{T})
    (F.N[1], F.N[2], F.N[3]÷F.num_processes)
end

function complex_shape{T<:Real}(F::r2c{T})
    (F.N[1]÷2+1, F.N[2]÷F.num_processes, F.N[3])
end

function complex_shape_T{T<:Real}(F::r2c{T})
    (F.N[1]÷2+1, F.N[2], F.N[3]÷F.num_processes)
end

function complex_local_slice{T<:Real}(F::r2c{T})
    ((1, F.N[1]÷2+1),
     (F.rank*F.N[2]÷F.num_processes+1, (F.rank+1)*F.N[2]÷F.num_processes),
     (1, F.N[3]))
end

function complex_local_wavenumbers{T<:Real}(F::r2c{T})
    (rfftfreq(F.N[1], 1.0/F.N[1]),
     fftfreq(F.N[2], 1.0/F.N[2])[F.rank*div(F.N[2], F.num_processes)+1:(F.rank+1)*div(F.N[2], F.num_processes)],
     fftfreq(F.N[3], 1.0/F.N[3]))
end

function get_local_wavenumbermesh{T<:Real}(F::r2c{T})
    K = Array{Int}(tuple(push!([complex_shape(F)...], 3)...))
    k = complex_local_wavenumbers(F)
    for (i, Ki) in enumerate(ndgrid(k[1], k[2], k[3])) K[view(i)...] = Ki end
    K
end

function get_local_mesh{T<:Real}(F::r2c{T})
    # Real grid
    x = collect(0:F.N[1]-1)*F.L[1]/F.N[1]
    y = collect(0:F.N[2]-1)*F.L[2]/F.N[2]
    z = collect(0:F.N[3]-1)*F.L[3]/F.N[3]
    X = Array{T}(tuple(push!([real_shape(F)...], 3)...))
    for (i, Xi) in enumerate(ndgrid(x, y, z[F.rank*F.N[3]÷F.num_processes+1:(F.rank+1)*F.N[3]÷F.num_processes])) X[view(i)...] = Xi end
    X
end

function dealias{T<:Real}(F::r2c{T}, fu::AbstractArray{Complex{T}, 3})
    kk = complex_local_wavenumbers(F)
    for (k, kz) in enumerate(kk[3])
        x = false
        if abs(kz) > div(F.N[3], 3)
        @inbounds fu[:, :, k] = 0.0
            continue
        end
        for (j, ky) in enumerate(kk[2])
            if abs(ky) > div(F.N[2], 3)
               @inbounds fu[:, j, k] = 0
                continue
            end
            for (i, kx) in enumerate(kk[1])
                if (abs(kx) > div(F.N[1], 3))
                    @inbounds fu[i, j, k] = 0.0
                end
            end
        end
    end
end

function dealias2{T<:Real}(F::r2c{T}, fu::AbstractArray{Complex{T}, 3})
    if  !isdefined(F, :dealias)
        const kmax_dealias = F.N/3
        K = get_local_wavenumbermesh(F)
        (kx, ky, kz) = K[:,:,:,1], K[:,:,:,2], K[:,:,:,3]
        indices = []
        i = 1
        for (x,y,z) in zip(kx, ky, kz)
            if abs(x) > div(F.N[1], 3) || abs(y) > div(F.N[2], 3) || abs(z) > div(F.N[3], 3)
                push!(indices, i)
            end
            i += 1
        end
        F.dealias = indices
    end
    for i in F.dealias
      @inbounds  fu[i] = 0.0
    end
end

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

# ----------------------------------------------------------------------------

using Base.LinAlg.BLAS: axpy!

function mhd(n)

    @assert n > 0 && (n & (n-1)) == 0 "n must be a power of 2"
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    num_processes = MPI.Comm_size(comm)

    const nu  = 0.1
    const eta = 0.1
    const dt  = 0.01
    const T   = 5.0
    const N = [n, n, n]    # Global shape of mesh
    const L = [2pi, 2pi, 2pi] # Real size of mesh

    FFT = r2c(N, L, comm)

    # DNS shapes
    rshape = real_shape(FFT)
    rvector_shape = tuple(push!([rshape...], 3)...)

    cshape = complex_shape(FFT)
    cvector_shape = tuple(push!([cshape...], 3)...)
    # MHD shapes
    rshapeMHD = real_shape(FFT)
    rvector_shapeMHD = tuple(push!([rshapeMHD...], 6)...)

    cshapeMHD = complex_shape(FFT)
    cvector_shapeMHD = tuple(push!([cshapeMHD...], 6)...)
    # MHD work array shapes
    rshapeMHD_tmp = real_shape(FFT)
    rvector_shapeMHD_tmp = tuple(push!([rshapeMHD_tmp...], 9)...)

    cshapeMHD_tmp = complex_shape(FFT)
    cvector_shapeMHD_tmp = tuple(push!([cshapeMHD_tmp...], 9)...)
    # Real vectors
    U_tmp = Array{Float64}(rvector_shape)
    B_tmp = Array{Float64}(rvector_shape)
    U = Array{Float64}(rvector_shapeMHD)
    F = Array{Float64}(rvector_shapeMHD_tmp)
    Z = similar(U)
    # Complex vectors
    dU = Array{Complex{Float64}}(cvector_shapeMHD)
    U_hat, U_hat0, U_hat1 = similar(dU), similar(dU), similar(dU)
    F_hat = Array{Complex{Float64}}(cvector_shapeMHD_tmp)
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

    "Elsasser vector Z"
    function ElsasserVector!(a, b)
        @assert size(a) == size(b)
        indices = linind(a)
        for axis in 1:last(size(a))-3
            iindexes = indices[axis]:indices[axis+1]-1
            jindexes = indices[axis+3]:indices[axis+4]-1
            @itr for (i, j) in zip(iindexes, jindexes)
                @inbounds b[i] = a[i] + a[j]
                @inbounds b[j] = a[i] - a[j]
            end
        end
    end
    "Product of Elsasser vectors"
    function ElsasserProduct!(a, b, c, FFT)
        indices = linind(b)
        tmp = 0
        for axis in 1:last(size(a))-3
            iindexes = indices[axis]:indices[axis+1]-1
            for yaxis in 1:last(size(a))-3
                jindexes = indices[yaxis+3]:indices[yaxis+1+3]-1
                kindexes = indices[yaxis+tmp]:indices[yaxis+1+tmp]-1
                @itr for (i, j, k) in zip(iindexes, jindexes, kindexes)
                    @inbounds b[k] = a[i]*a[j]
                end
            end
            tmp += 3
        end
        for i in 1:9 rfft3(FFT, c(i), b[view(i)...]) end
    end
    "Divergence convection"
    function DivergenceConvection!(K, a, b)
        indices = linind(a)
        kx_indices = indices[1]:indices[2]-1
        ky_indices = indices[2]:indices[3]-1
        kz_indices = indices[3]:indices[4]-1
        tmp = 0
        for axis in 1:last(size(K))
            i_indices  = indices[axis]:indices[axis+1]-1
            i3_indices = indices[axis+3]:indices[axis+4]-1
            i6_indices = indices[axis+6]:indices[axis+7]-1

            itmp0_indices  = indices[axis+tmp]:indices[axis+tmp+1]-1
            itmp1_indices = indices[axis+1+tmp]:indices[axis+2+tmp]-1
            itmp2_indices = indices[axis+2+tmp]:indices[axis+3+tmp]-1

            @itr for (i, i3, i6, tmp0, tmp1, tmp2, k1, k2, k3) in zip(i_indices, i3_indices, i6_indices, itmp0_indices, itmp1_indices, itmp2_indices, kx_indices, ky_indices, kz_indices)
                    @inbounds b[i] = -0.5im*(K[k1]*(a[i] + a[tmp0]) + K[k2]*(a[i3] + a[tmp1]) + K[k3]*(a[i6] + a[tmp2]))
                    @inbounds b[i3] = 0.5im *(K[k1]*(a[i] - a[tmp0]) + K[k2]*(a[i3] - a[tmp1]) + K[k3]*(a[i6] - a[tmp2]))
            end
            tmp += 2
        end
    end

    "sources, rk, out"
    function ComputeRHS!(wcross, wcurl, U, U_hat, Z, F, F_hat, K, K_over_K2, K2, P_hat, nu, eta, rk, dU, FFT)
        for i in 1:6 irfft3(FFT, U(i), U_hat[view(i)...], 1) end

        # Construct the Elsasser vector
        ElsasserVector!(U, Z)
        ElsasserProduct!(Z, F, F_hat, FFT)
        DivergenceConvection!(K, F_hat, dU)

        # Add pressure gradient
        P_hat[:] = zero(eltype(P_hat))
        indices = linind(dU)
        for axis in 1:last(size(dU))-3
            @itr for (j, i) in enumerate(indices[axis]:indices[axis+1]-1)
                @inbounds P_hat[j] += dU[i]*K_over_K2[i]
            end
        end
        # Add diffusion
        for axis in 1:last(size(dU))
            @itr for (j, i) in enumerate(indices[axis]:indices[axis+1]-1)
                if axis < 4
                    @inbounds dU[i] -= P_hat[j]*K[i] + nu*U_hat[i]*K2[j]
                else
                    @inbounds dU[i] -= eta*U_hat[i]*K2[j]
                end
            end
        end
    end

    U[view(1)...] = sin(X(1)).*cos(X(2)).*cos(X(3))
    U[view(2)...] = -cos(X(1)).*sin(X(2)).*cos(X(3))
    U[view(3)...] = 0.

    U[view(4)...] = sin(X(1)).*sin(X(2)).*cos(X(3))
    U[view(5)...] = cos(X(1)).*cos(X(2)).*cos(X(3))
    U[view(6)...] = 0.

    #  taking fft
    for i in 1:6 rfft3(FFT,  U_hat(i), U(i)) end

    t = 0.0
    tstep = 0
    Ek = []
    Eb = []
    t_vec = []
    while t < T-1e-8

        t += dt; tstep += 1
        U_hat1[:] = U_hat; U_hat0[:] = U_hat

        for rk in 1:4
            ComputeRHS!(wcross, wcurl, U, U_hat, Z, F, F_hat, K, K_over_K2, K2, P_hat, nu, eta, rk, dU, FFT)
            if rk < 4
                U_hat[:] = U_hat0
                axpy!(b[rk], dU, U_hat)
            end
            axpy!(a[rk], dU, U_hat1)
        end
        U_hat[:] = U_hat1
        for i in 1:6 irfft3(FFT, U(i), U_hat[view(i)...]) end

        for i in 1:3
            U_tmp[view(i)...] = U(i)
            B_tmp[view(i)...] = U(i+3)
        end
        ek = MPI.Reduce(0.5*sumabs2(U_tmp)*(1./prod(FFT.N)), MPI.SUM, 0, comm)
        eb = MPI.Reduce(0.5*sumabs2(B_tmp)*(1./prod(FFT.N)), MPI.SUM, 0, comm)
        if rank == 0
            push!(Ek, ek)
            push!(Eb, eb)
            push!(t_vec, t)
        end
    end
    MPI.Finalize()
end
mhd(32)
