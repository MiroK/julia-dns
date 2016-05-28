function view{T, N}(A::Array{T, N}, index::Integer)
    dims = size(A)
    @assert 1 <= index <= last(dims)
    index = [fill(Colon(), length(dims)-1);
