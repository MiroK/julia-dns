
#=
Created on Fri 19 Aug 20:23:35 2016

@author: Diako Darian

3D periodic MHD-solver using Fourier-Galerkin method
=#

using PyCall
import IJulia
using PyPlot
using Iterators

include("utils.jl")
using Utils  
using Compat

view(k::Int, N::Int=4) = [fill(Colon(), N-1)..., k]

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

# ----------------------------------------------------------------------------

using Base.LinAlg.BLAS: axpy!

function mhd(N)
    @assert N > 0 && (N & (N-1)) == 0 "N must be a power of 2"

    const nu  = 0.01
    const eta = 0.01
    const dt  = 0.01
    const T   = 1.0
    const Nh  = N÷2+1

    # Real vectors
    U_tmp = Array{Float64}(N, N, N, 3)
    B_tmp = Array{Float64}(N, N, N, 3)
    U = Array{Float64}(N, N, N, 6)
    F = Array{Float64}(N, N, N , 9)
    Z = similar(U)
    # Complex vectors
    dU = Array{Complex{Float64}}(Nh, N, N, 6)
    U_hat, U_hat0, U_hat1 = similar(dU), similar(dU), similar(dU)
    F_hat = Array{Complex{Float64}}(Nh, N, N, 9)
    # Complex scalars
    P_hat = Array{eltype(dU)}(Nh, N, N)

    # Real grid
    x = collect(0:N-1)*2*pi/N
    X = similar(U)
    for (i, Xi) in enumerate(ndgrid(x, x, x)) X[view(i)...] = Xi end
    # Wave number grid
    kx = fftfreq(N, 1./N)
    kz = kx[1:(N÷2+1)]; kz[end] *= -1
    K = Array{eltype(U)}(Nh, N, N, 3)
    for (i, Ki) in enumerate(ndgrid(kz, kx, kx)) K[view(i)...] = Ki end
    # Square of wave number vectors
    K2 = Array{eltype(U)}(Nh, N, N)
    sumabs2!(K2, K)
    # K/K2
    K_over_K2 = K./K2             
    # Fix division by zero
    for i in 1:3 K_over_K2[1, 1, 1, i] = K[1, 1, 1, i] end
      
    # Dealising mask
    const kmax_dealias = 2*Nh/3
    dealias = reshape(reduce(&, [abs(K(i)) .< kmax_dealias for i in 1:3]), Nh, N, N)
    # Runge-Kutta weights
    a = dt*[1./6., 1./3., 1./3., 1./6.]  
    b = dt*[0.5, 0.5, 1.] 
    # Work arrays for cross
    wcross = Array{eltype(U)}(N, N, N)
    wcurl = Array{eltype(dU)}(Nh, N, N)

    # Define (I)RFFTs
    const RFFT = plan_rfft(wcross, (1, 2, 3))
    fftn_mpi!(u, fu) = A_mul_B!(fu, RFFT, u)

    const IRFFT = plan_irfft(wcurl, N, (1, 2, 3))
    ifftn_mpi!(fu, u) = A_mul_B!(u, IRFFT, fu)

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
    function ElsasserProduct!(a, b, c)
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
        for i in 1:9 fftn_mpi!(b[view(i)...], c(i)) end
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

    function ComputeRHS!(wcross, wcurl, U, U_hat, Z, F, F_hat, K, K_over_K2, K2, P_hat, nu, eta, rk, dU)
        for i in 1:6
            broadcast!(*, wcurl, U_hat[view(i)...], dealias)  # Use wcurl as work array
            ifftn_mpi!(wcurl, U(i))
        end
        
        # Construct the Elsasser vector
        ElsasserVector!(U, Z)
        ElsasserProduct!(Z, F, F_hat)
        DivergenceConvection!(K, F_hat, dU)

        indices = linind(dU)
        # Add pressure gradient
        P_hat[:] = zero(eltype(P_hat))
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

    for i in 1:6 fftn_mpi!(U(i), U_hat(i)) end

    t = 0.0
    tstep = 0
    Ek = []
    Eb = []
    t_vec = []
    while t < T-1e-8

        t += dt; tstep += 1
        U_hat1[:] = U_hat; U_hat0[:] = U_hat
        
        for rk in 1:4
            ComputeRHS!(wcross, wcurl, U, U_hat, Z, F, F_hat, K, K_over_K2, K2, P_hat, nu, eta, rk, dU)
            if rk < 4
                U_hat[:] = U_hat0
                axpy!(b[rk], dU, U_hat)
            end
            axpy!(a[rk], dU, U_hat1)
        end

        U_hat[:] = U_hat1
        for i in 1:6 ifftn_mpi!(U_hat[view(i)...], U(i)) end
        
        for i in 1:3
            U_tmp[view(i)...] = U(i)
            B_tmp[view(i)...] = U(i+3)
        end
        ek = 0.5*sumabs2(U_tmp)*(1./N)^3
        eb = 0.5*sumabs2(B_tmp)*(1./N)^3
        push!(Ek, ek)
        push!(Eb, eb)
        push!(t_vec, t)
    end

    fig = figure("Kinetic and magnetic energies",figsize=(8,6))

    subplot(211)
    plot(t_vec, Ek)
    ax = gca()
    axis("tight")
    ax[:spines]["top"][:set_visible](false) # Hide the top edge of the axis
    ax[:spines]["right"][:set_visible](false) # Hide the right edge of the axis
    ax[:xaxis][:set_ticks_position]("bottom")
    ax[:yaxis][:set_ticks_position]("left")
    ax[:spines]["left"][:set_position](("axes",-0.03)) # Offset the left scale from the axis
    ax[:spines]["bottom"][:set_position](("axes",-0.05)) # Offset the bottom scale from the axis
    grid("on")
    xlabel("Time")
    ylabel("Kinetic energy")
    

    subplot(212)
    plot(t_vec, Eb)
    ax2 = gca()
    axis("tight")
    ax2[:spines]["top"][:set_visible](false) # Hide the top edge of the axis
    ax2[:spines]["right"][:set_visible](false) # Hide the right edge of the axis
    ax2[:xaxis][:set_ticks_position]("bottom")
    ax2[:yaxis][:set_ticks_position]("left")
    ax2[:spines]["left"][:set_position](("axes",-0.03)) # Offset the left scale from the axis
    ax2[:spines]["bottom"][:set_position](("axes",-0.03)) # Offset the bottom scale from the axis
    grid("on")
    xlabel("Time")
    ylabel("Magnetic energy")
end
