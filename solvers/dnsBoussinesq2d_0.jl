#=
Created on Fri 19 Aug 11:28:46 2016

@author: Diako Darian

2-dimensional Navier-Stokes equation with boussinesq approximation
Solved using Fourier-Galerkin method

=#

using PyCall
import IJulia
@pyimport matplotlib.animation as animation
using PyPlot


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
    Pr = 4.0
    Ri = 0.167
    dt = 0.01
    T = 5.0
    Nh = N÷2+1
    plot_result = 20
    "Kelvin-Helmholtz parameters:"
    U1 = -0.5
    U2 = 0.5
    A  = 0.01
    rho1 = 1.0
    rho2 = 3.0 
    delta = 0.05

    # Real vectors
    Ur    = Array[RArray(N, N) for i in 1:3]
    # Complex vectors
    Ur_hat  = Array[CArray(Nh, N) for i in 1:3]
    Ur_hat0 = Array[CArray(Nh, N) for i in 1:3]
    Ur_hat1 = Array[CArray(Nh, N) for i in 1:3]
    dU      = Array[CArray(Nh, N) for i in 1:3]
    conv    = Array[CArray(Nh, N) for i in 1:2]
    # Complex scalars
    Uc_hat = CArray(Nh, N)
    P_hat  = CArray(Nh, N)
    # Transpose
    Uc_hatT = CArray(Nh, N)
    # Real scalars
    P = RArray(N, N)
    curl = RArray(N, N) 
    
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
    function ComputeRHS!(Ur, Ur_hat, curl, K, K_over_K2, K2, P_hat, nu, Pr, Ri, rk, dU)
        if rk > 1
            for i in 1:3 ifft2_mpi!(Ur_hat[i], Ur[i]) end
        end
        # Convective term in N-S-equation
        Curl2!(Ur_hat, K, curl)
        fft2_mpi!(Ur[2].*curl, dU[1])
        fft2_mpi!(-Ur[1].*curl, dU[2])
        # Convective term in density equation
        for i in 1:2 fft2_mpi!(Ur[i].*Ur[3], conv[i]) end
        dU[3] = -1.0im*(K[1].*conv[1] + K[2].*conv[2])
        # Dealias the convective terms
        for i in 1:3 dU[i] .*= dealias end
        # Calculate the pressure
        P_hat[:] = reduce(+, [dU[i].*K_over_K2[i] for i in 1:2])
        P_hat -= Ri*Ur_hat[3].*K_over_K2[2]
        # Add the pressure gradient to momentum eq.
        for i in 1:2 dU[i] -= P_hat.*K[i] end
        # Add the diffusion terms
        dU[1] -= nu*K2.*Ur_hat[1]
        dU[2] -= (nu*K2.*Ur_hat[2] + Ri*Ur_hat[3])
        dU[3] -= nu*K2.*Ur_hat[3]/Pr
    end

    Um = 0.5*(U1 - U2)
    rho0 = 0.5*(rho1 + rho2)

    copy!(Ur[2], A*sin(2*X[1]))
    
    for i in 1:N
        for j in 1:N
            if j<div(N,2)
                Ur[1][i,j] = tanh((X[2][i,j] -0.5*pi)/delta)    
                Ur[3][i,j] = 2.0-rho0 + tanh((X[2][i,j] -0.5*pi)/delta)
            else
                Ur[1][i,j] = -tanh((X[2][i,j] -1.5*pi)/delta)    
                Ur[3][i,j] = 2.0-rho0 - tanh((X[2][i,j] -1.5*pi)/delta)
            end
        end
    end
    #copy!(Ur[1][:, 1:div(N,2)], tanh((X[2][:, 1:div(N,2)] -0.5*pi)/delta))
    #copy!(Ur[1][:, div(N,2):end], -tanh((X[2][:, div(N,2):end]-1.5*pi)/delta))
    #copy!(Ur[3][:, 1:div(N,2)], 2.0 + tanh((X[2][:, 1:div(N,2)] -0.5*pi)/delta))
    #copy!(Ur[3][:, div(N,2):end], 2.0 -tanh((X[2][:, div(N,2):end]-1.5*pi)/delta)) 
    #Ur[3] -= rho0

    #-------------------------------------------------------------------------------
    # Plots
    #-------------------------------------------------------------------------------
    fig = figure("pyplot_imshow",figsize=(12,8))
    title("rho", fontsize=20)
    xlabel("x")
    ylabel("y")

    get_cmap("RdBu")
    image = imshow(Ur[3],cmap="RdBu", extent=[0, 2*pi, 0, 2*pi])
    colorbar(image)
    draw()
    #-------------------------------------------------------------------------------

    for i in 1:3 fft2_mpi!(Ur[i], Ur_hat[i]) end

    t = 0.0
    tstep = 0
    while t < T-1e-8
        t += dt; tstep += 1
        Ur_hat1[:] = Ur_hat[:]; Ur_hat0[:]=Ur_hat[:]
        
        for rk in 1:4
            ComputeRHS!(Ur, Ur_hat, curl, K, K_over_K2, K2, P_hat, nu, Pr, Ri, rk, dU)
            if rk < 4 Ur_hat[:] = Ur_hat0[:] + b[rk]*dU[:] end
            for i in 1:3 Ur_hat1[i] += a[rk]*dU[i] end
        end

        copy!(Ur_hat, Ur_hat1)
        for i in 1:3 ifft2_mpi!(Ur_hat[i], Ur[i]) end
        #-------------------------------------------------------------------------------
        # Plots
        #-------------------------------------------------------------------------------
        if tstep % plot_result == 0
            println(tstep)
            imshow(Ur[3],cmap="RdBu", extent=[0, 2*pi, 0, 2*pi])
            #image[set_data(Ur[3])]
            #image[autoscale()]
            pause(1e-6)
       end
       #-------------------------------------------------------------------------------
    end
end
