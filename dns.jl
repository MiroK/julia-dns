include("utils.jl")
using Utils

function dns(N)
    nu = 0.000625
    T = 0.1
    dt = 0.01
    Nh = N÷2+1

    # Real vectors
    U    = Array[RArray(N, N, N) for i in 1:3]
    curl = Array[RArray(N, N, N) for i in 1:3]
    # Complex vectors
    U_hat  = Array[CArray(Nh, N, N) for i in 1:3]
    U_hat0 = Array[CArray(Nh, N, N) for i in 1:3]
    U_hat1 = Array[CArray(Nh, N, N) for i in 1:3]
    dU     = Array[CArray(Nh, N, N) for i in 1:3]
    # Complex scalars
    Uc_hat = CArray(Nh, N, N)
    P_hat  = CArray(Nh, N, N)
    # Transpose
    Uc_hatT = CArray(Nh, N, N)
    # Real scalars
    P = RArray(N, N, N)
    # Real grid
    x = collect(0:N-1)*2*pi/N
    X = Array[ndgrid(x, x, x)...]
    # Complex grid
    kx = fftfreq(N, 1./N)
    kz = kx[1:(N÷2+1)]; kz[end] *= -1
    K = Array[ndgrid(kz, kx, kx)...]
    # Square of wave number vectors
    K2 = K[1].^2 + K[2].^2 + K[3].^2
    # K/K2 term
    K_over_K2 = Array[K[i]./K2 for i in 1:3]
    # Fix division by zero
    for i in 1:3 K_over_K2[i][1] = K[i][1] end
    # Dealising mask
    kmax_dealias = 2*Nh/3
    dealias = (abs(K[1]).<kmax_dealias).*(abs(K[2]).<kmax_dealias).*(abs(K[3]).<kmax_dealias)
    a = [1./6., 1./3., 1./3., 1./6.]  
    b = [0.5, 0.5, 1.]              

    function Cross!(a, b, c)
        fftn_mpi!(a[2].*b[3]-a[3].*b[2], c[1])
        fftn_mpi!(a[3].*b[1]-a[1].*b[3], c[2])
        fftn_mpi!(a[1].*b[2]-a[2].*b[1], c[3])
    end

    function Curl!(a, K, c)
        ifftn_mpi!(im*(K[1].*a[2]-K[2].*a[1]), c[3])
        ifftn_mpi!(im*(K[3].*a[1]-K[1].*a[3]), c[2])
        ifftn_mpi!(im*(K[2].*a[3]-K[3].*a[2]), c[1])
    end

    "sources, rk, out"
    function ComputeRHS!(U, U_hat, curl, K, K_over_K2, K2, P_hat, nu, rk, dU)
        if rk > 1
            for i in 1:3 ifftn_mpi!(U_hat[i], U[i]) end
        end
        Curl!(U_hat, K, curl)
        Cross!(U, curl, dU)
        for i in 1:3 dU[i] .*= dealias end
        
        copy!(P_hat, dU[1].*K_over_K2[1]); for i in 2:3 P_hat += dU[i].*K_over_K2[i] end
        for i in 1:3 dU[i] -= P_hat.*K[i] end
        for i in 1:3 dU[i] -= nu*K2.*U_hat[i] end
    end

    U[1] = sin(X[1]).*cos(X[2]).*cos(X[3])
    U[2] =-cos(X[1]).*sin(X[2]).*cos(X[3])
    setindex!(U[3], 0, :)

    for i in 1:3 fftn_mpi!(U[i], U_hat[i]) end

    t = 0.0
    tstep = 0
    while t < T-1e-8
        t += dt; tstep += 1
        copy!(U_hat1, U_hat); copy!(U_hat0, U_hat)
        
        for rk in 1:4
            ComputeRHS!(U, U_hat, curl, K, K_over_K2, K2, P_hat, nu, rk, dU)
            if rk < 4 
                for i in 1:3 copy!(U_hat[i], U_hat0[i] + b[rk]*dt*dU[i]) end
            end
            for i in 1:3 U_hat1[i] += a[rk]*dt*dU[i] end
        end

        copy!(U_hat, U_hat1)
        for i in 1:3 ifftn_mpi!(U_hat[i], U[i]) end
    end
     
    k = 0.5*sum(U[1].*U[1]+U[2].*U[2]+U[3].*U[3])*(1./N)^3
    
    U
end
