using MPI

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

    # Constructor
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


function foo()
    const comm = MPI.COMM_WORLD
    const rank = MPI.Comm_rank(comm)
    const num_processes = MPI.Comm_size(comm)
    const N = 64    
    const Nh = N÷2+1
    const Np = N÷num_processes

    # Real vectors
    U = rand(Float64, N, N, Np)
    fU = Array{Complex{Float64}}(Nh, Np, N)
    ffU = similar(U)

    const F = SpecTransf(U, comm)

    for i in 1:60
        apply(fU, F, U)
        apply_inv(ffU, F, fU)
    end

    error = MPI.Reduce(sumabs2(ffU-U), MPI.SUM, 0, comm) 
    error
end

MPI.Init()
println(foo())
@time foo()
MPI.Finalize()
