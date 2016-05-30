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

"fftn from dns.py"
fftn_mpi!(u, fu) = copy!(fu, rfft(u, (1, 2, 3)))
"ifftn from dns.py"
ifftn_mpi!(fu, u) = copy!(u, irfft(fu, first(size(u)), (1, 2, 3)))

# In dns we use real and complex arrays. Declaring these types here makes it
# easier to propagate changes.
typealias RealT Float64
typealias CmplT Complex128
typealias RArray Array{RealT}
typealias CArray Array{CmplT}

export ndgrid, fftfreq, RealT, CmplT, RArray, CArray, fftn_mpi!, ifftn_mpi!

end
