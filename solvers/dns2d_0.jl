# 2-dimensional DNS-solver using Fourier-Galerkin method

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

fft2_mpi!(u, fu) = copy!(fu, rfft(u, (1, 2)))
ifft2_mpi!(fu, u) = copy!(u, irfft(fu, first(size(u)), (1, 2)))

typealias RealT Float64
typealias CmplT Complex128
typealias RArray Array{RealT}
typealias CArray Array{CmplT}

function dns(N)
    @assert N > 0 && (N & (N-1)) == 0 "N must be a power of 2"

    nu = 0.000625
    dt = 0.01
    T = 0.1
    Nh = N÷2+1

    # Real vectors
    U    = Array[RArray(N, N) for i in 1:2]
    curl = RArray(N, N) 
    # Complex vectors
    U_hat  = Array[CArray(Nh, N) for i in 1:2]
    U_hat0 = Array[CArray(Nh, N) for i in 1:2]
    U_hat1 = Array[CArray(Nh, N) for i in 1:2]
    dU     = Array[CArray(Nh, N) for i in 1:2]
    # Complex scalars
    Uc_hat = CArray(Nh, N)
    P_hat  = CArray(Nh, N)
    # Transpose
    Uc_hatT = CArray(Nh, N)
    # Real scalars
    P = RArray(N, N)
    
    # Real grid
    x = collect(0:N-1)*2*pi/N
    X = Array[ndgrid(x, x)...]
    # Complex grid
    kx = fftfreq(N, 1./N)
    ky = kx[1:(N÷2+1)]; ky[end] *= -1
    K = Array[ndgrid(ky, kx)...]
    # Square of wave number vectors
    K2 = K[1].^2 + K[2].^2 
    # K/K2 term
    K_over_K2 = Array[K[i]./K2 for i in 1:2]
    # Fix division by zero
    for i in 1:2 K_over_K2[i][1] = K[i][1] end
    # Dealising mask
    kmax_dealias = 2*Nh/3
    dealias = (abs(K[1]).<kmax_dealias).*(abs(K[2]).<kmax_dealias)
    a = dt*[1./6., 1./3., 1./3., 1./6.]  
    b = dt*[0.5, 0.5, 1.]              

    function Curl2!(a, K, c)
        ifft2_mpi!(1.0im*(K[1].*a[2]-K[2].*a[1]), c)
    end

    "sources, rk, out"
    function ComputeRHS!(U, U_hat, curl, K, K_over_K2, K2, P_hat, nu, rk, dU, S, X, t)
        if rk > 1
            for i in 1:2 ifft2_mpi!(U_hat[i], U[i]) end
        end
 
        Curl2!(U_hat, K, curl)
        fft2_mpi!(U[2].*curl, dU[1])
        fft2_mpi!(-U[1].*curl, dU[2])

        for i in 1:2 dU[i] .*= dealias end

        P_hat[:] = reduce(+, [dU[i].*K_over_K2[i] for i in 1:2])

        for i in 1:2 dU[i] -= P_hat.*K[i] end
        for i in 1:2 dU[i] -= nu*K2.*U_hat[i] end
    end

    copy!(U[1], sin(X[1]).*cos(X[2]))
    copy!(U[2],-cos(X[1]).*sin(X[2]))
 
    for i in 1:2 fft2_mpi!(U[i], U_hat[i]) end

    t = 0.0
    tstep = 0
    t_min, t_max = NaN, 0
    while t < T-1e-8
        tic()
        
        t += dt; tstep += 1
        U_hat1[:] = U_hat[:]; U_hat0[:]=U_hat[:]
        
        for rk in 1:4
            ComputeRHS!(U, U_hat, curl, K, K_over_K2, K2, P_hat, nu, rk, dU, S, X, t)
            if rk < 4 U_hat[:] = U_hat0[:] + b[rk]*dU[:] end
            for i in 1:2 U_hat1[i] += a[rk]*dU[i] end
        end

        copy!(U_hat, U_hat1)
        for i in 1:2 ifft2_mpi!(U_hat[i], U[i]) end
        
        time_step = toq()
        t_min = min(time_step, t_min)
        t_max = max(time_step, t_max)
    end
    
    k = 0.5*sum(U[1].*U[1]+U[2].*U[2])*(1./N)^2
    println((k, t_min, t_max))
end
