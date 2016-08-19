
#=
Created on Fri 19 Aug 11:28:46 2016

@author: Diako Darian

2-dimensional DNS-solver using Fourier-Galerkin method
Linear indexing is used for calulations.
=#

using PyCall
import IJulia
@pyimport matplotlib.animation as animation
using PyPlot

include("utils.jl")
using Utils  
using Compat

view(k::Int, N::Int=3) = [fill(Colon(), N-1)..., k]

"View of A with last coordinate fixed at k"
@compat function call{T, N}(A::Array{T, N}, k::Int)
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
function cross!{S, T, R}(X::AbstractArray{S, 3},
                         Y::AbstractArray{T, 3},
                         w::AbstractArray{R, 2})
    @assert size(X) == size(Y) 

    indices = linind(X)
    iaxis, jaxis = 1, 2 
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

    const nu = 0.000625
    const dt = 0.01
    const T = 0.1
    const Nh = N÷2+1
    const plot_result = 1

    # Real vectors
    U = Array{Float64}(N, N, 2)
    F = similar(U)
    # Complex vectors
    dU = Array{Complex{Float64}}(Nh, N, 2)
    U_hat, U_hat0, U_hat1 = similar(dU), similar(dU), similar(dU)
    # Complex scalars
    P_hat = Array{eltype(dU)}(Nh, N)
    # Real scalars
    curl = Array{eltype(U)}(N,N)
    # Real grid
    x = collect(0:N-1)*2*pi/N
    X = similar(U)
    for (i, Xi) in enumerate(ndgrid(x, x)) X[view(i)...] = Xi end
    # Wave number grid
    kx = fftfreq(N, 1./N)
    kz = kx[1:(N÷2+1)]; kz[end] *= -1
    K = Array{eltype(U)}(Nh, N, 2)
    for (i, Ki) in enumerate(ndgrid(kz, kx)) K[view(i)...] = Ki end
    # Square of wave number vectors
    K2 = Array{eltype(U)}(Nh, N)
    sumabs2!(K2, K)
    # K/K2
    K_over_K2 = K./K2             
    # Fix division by zero
    for i in 1:2 K_over_K2[1, 1, i] = K[1, 1, i] end
    # Dealising mask
    const kmax_dealias = 2*Nh/3
    dealias = reshape(reduce(&, [abs(K(i)) .< kmax_dealias for i in 1:2]), Nh, N)
    # Runge-Kutta weights
    a = dt*[1./6., 1./3., 1./3., 1./6.]  
    b = dt*[0.5, 0.5, 1.] 
    # Work arrays for cross
    wcross = Array{eltype(U)}(N, N)
    wcurl = Array{eltype(dU)}(Nh, N)

    # Define (I)RFFTs
    const RFFT = plan_rfft(wcross, (1, 2))
    fftn_mpi!(u, fu) = A_mul_B!(fu, RFFT, u)

    const IRFFT = plan_irfft(wcurl, N, (1, 2))
    ifftn_mpi!(fu, u) = A_mul_B!(u, IRFFT, fu)

    function Cross!(w, a, b, c)
        for i in 1:3
            cross!(i, a, b, w)
            fftn_mpi!(w, c(i))
        end
    end

    function Curl!(w, a, K, c, dealias)
        cross!(K, a, w)
        scale!(w, im)
        broadcast!(*, w, w, dealias) 
        ifftn_mpi!(w, c)
    end

    function ComputeRHS!(wcross, wcurl, U, U_hat, curl, K, K_over_K2, K2, P_hat, nu, rk, dU)
        for i in 1:2 
            broadcast!(*, wcurl, U_hat[view(i)...], dealias)  # Use wcurl as work array
            ifftn_mpi!(wcurl, U(i))
        end

        indices = linind(dU)

        Curl!(wcurl, U_hat, K, curl, dealias)
        F[:] = zero(eltype(F))
        for axis in 1:last(size(U))
            for (j, i) in enumerate(indices[axis]:indices[axis+1]-1)
                @inbounds F[i] = curl[j]*U[i]
            end
        end
        fftn_mpi!(F(2), dU(1))
        fftn_mpi!(-1.0*F(1), dU(2))
        #Cross!(wcross, U, curl, dU)

        P_hat[:] = zero(eltype(P_hat))
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

    U[view(1)...] = sin(X(1)).*cos(X(2))
    U[view(2)...] = -cos(X(1)).*sin(X(2))

    #-------------------------------------------------------------------------------
    # Plots
    #-------------------------------------------------------------------------------
    fig = figure("pyplot_imshow",figsize=(12,8))
    title("U[1]", fontsize=20)
    xlabel("x")
    ylabel("y")

    get_cmap("RdBu")
    image = imshow(U(1),cmap="RdBu", extent=[0, 2*pi, 0, 2*pi])
    colorbar(image)
    draw()
    #-------------------------------------------------------------------------------

    for i in 1:2 fftn_mpi!(U(i), U_hat(i)) end

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
        for i in 1:2 ifftn_mpi!(U_hat[view(i)...], U(i)) end
        #-------------------------------------------------------------------------------
        # Plots
        #-------------------------------------------------------------------------------
        if tstep % plot_result == 0
            println(tstep)
            imshow(U(1),cmap="RdBu", extent=[0, 2*pi, 0, 2*pi])
            #image[set_data(Ur[3])]
            #image[autoscale()]
            pause(1e-6)
       end
       #-------------------------------------------------------------------------------
        time_step = toq()
        t_min = min(time_step, t_min)
        t_max = max(time_step, t_max)
    end

    for i in 1:2 ifftn_mpi!(U_hat[view(i)...], U(i)) end
    k = 0.5*sumabs2(U)*(1./N)^2
    (k, t_min)  
    println((tstep, k, t_min, t_max))
end
