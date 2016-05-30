# This is the starting point. Faster but memory hungry. For the sake of being
# self contained the utils module is replicated here.

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

"fftn from dns.py"
fftn_mpi!(u, fu) = fu[:] = rfft(u, (1, 2, 3))

"ifftn from dns.py"
ifftn_mpi!(fu, u) = u[:] = irfft(fu, first(size(u)), (1, 2, 3))

"View of A along the last axis"
function _{T, N}(A::AbstractArray{T, N}, k::Integer)
   @assert 1 <= k <= last(size(A))
   indices = [fill(Colon(), N-1)..., k]
   slice(A, indices...)
end

"Indexes for viewing into last axis of 4d array"
view(k::Integer) = (Colon(), Colon(), Colon(), k)
   
# ----------------------------------------------------------------------------

using Base.LinAlg.BLAS: axpy!
#N = 2^5
function dns(N)
    const nu = 0.000625
    const dt = 0.01
    const T = 0.1
    const Nh = N÷2+1

    # Real vectors
    U = Array{Float64}(N, N, N, 3)
    curl = similar(U)
    # Complex vectors
    dU = Array{Complex128}(Nh, N, N, 3)
    U_hat, U_hat0, U_hat1  = similar(dU), similar(dU), similar(dU)
    # Complex scalars
    P_hat = Array{Complex128}(Nh, N, N)
    # Real grid
    x = collect(0:N-1)*2*pi/N
    X = similar(U)
    for (i, Xi) in enumerate(ndgrid(x, x, x)) X[view(i)...] = Xi end
    # Wave number grid
    kx = fftfreq(N, 1./N)
    kz = kx[1:(N÷2+1)]; kz[end] *= -1
    K = Array{Float64}(Nh, N, N, 3)
    for (i, Ki) in enumerate(ndgrid(kz, kx, kx)) K[view(i)...] = Ki end
    # Square of wave number vectors
    K2 = sumabs2(K, 4)
    # K/K2
    K_over_K2 = K./K2
    # Fix division by zero
    for i in 1:3 K_over_K2[1, 1, 1, i] = K[1, 1, 1, i] end
    # Dealising mask
    const kmax_dealias = 2*Nh/3
    dealias = reshape(reduce(&, [abs(_(K, i)) .< kmax_dealias for i in 1:3]), Nh, N, N, 1)
    # Runge-Kutta weights
    a = dt*[1./6., 1./3., 1./3., 1./6.]  
    b = dt*[0.5, 0.5, 1.] 

    function Cross!(a, b, c)
        fftn_mpi!(_(a, 2).*_(b, 3)-_(a, 3).*_(b, 2), _(c, 1))
        fftn_mpi!(_(a, 3).*_(b, 1)-_(a, 1).*_(b, 3), _(c, 2))
        fftn_mpi!(_(a, 1).*_(b, 2)-_(a, 2).*_(b, 1), _(c, 3))
    end

    function Curl!(a, K, c)
        ifftn_mpi!(im*(_(K, 1).*_(a, 2)-_(K, 2).*_(a, 1)), _(c, 3))
        ifftn_mpi!(im*(_(K, 3).*_(a, 1)-_(K, 1).*_(a, 3)), _(c, 2))
        ifftn_mpi!(im*(_(K, 2).*_(a, 3)-_(K, 3).*_(a, 2)), _(c, 1))
    end

    "sources, rk, out"
    function ComputeRHS!(U, U_hat, curl, K, K_over_K2, K2, P_hat, nu, rk, dU)
        if rk > 1
            for i in 1:3 ifftn_mpi!(_(U_hat, i), _(U, i)) end
        end

        Curl!(U_hat, K, curl)
        Cross!(U, curl, dU)
        dU[:] = dU .* dealias

        P_hat[:] = sum(dU.*K_over_K2, 4)

        axpy!(-1., P_hat.*K, dU)
        axpy!(-1., nu*U_hat.*K2, dU)
    end

    U[view(1)...] = sin(_(X, 1)).*cos(_(X, 2)).*cos(_(X, 3))
    U[view(2)...] = -cos(_(X, 1)).*sin(_(X, 2)).*cos(_(X, 3))
    U[view(3)...] = 0.

    for i in 1:3 fftn_mpi!(_(U, i), _(U_hat, i)) end

    t = 0.0
    tstep = 0
    while t < T-1e-8
        t += dt; tstep += 1
        U_hat1[:] = U_hat; U_hat0[:] = U_hat
        
        for rk in 1:4
            ComputeRHS!(U, U_hat, curl, K, K_over_K2, K2, P_hat, nu, rk, dU)
            if rk < 4
                U_hat[:] = U_hat0
                axpy!(b[rk], dU, U_hat)
            end
            axpy!(a[rk], dU, U_hat1)
        end
        U_hat[:] = U_hat1
        for i in 1:3 ifftn_mpi!(_(U_hat, i), _(U, i)) end

    end

    k = 0.5*sumabs2(U)*(1./N)^3
    k
end
