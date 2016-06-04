# These functions are needed by dns.jl but are not part of julia.Base.
# Function ndgrid is taken from julia examples.
# fftfreq is implemented following scipy/numpy.

module Utils

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

export ndgrid, fftfreq, @mpitime

end
