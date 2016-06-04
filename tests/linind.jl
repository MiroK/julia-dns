function linind{T, N}(A::AbstractArray{T, N})
    L = prod(size(A)[1:N-1])
    indices = [1]
    for k in 1:size(A, N) push!(indices, last(indices)+L) end
    indices
end

function Linind{T, N}(A::AbstractArray{T, N})
    s = size(A)
    L = prod(s[1:N-1])
    indices = [1; zeros(Int, s[end])]
    for k in 1:size(A, N) indices[k+1] = indices[k] + L end
    indices
end

function foo(A)
    indices = linind(A)
    reduce(+, [sum(A[indices[k]:indices[k+1]-1]) for k in 1:last(size(A))])
end

function bar(A)
    indices = Linind(A)
    reduce(+, [sum(A[indices[k]:indices[k+1]-1]) for k in 1:last(size(A))])
end


A = rand(64, 64, 64, 3)
const INDICES = linind(A)

function baz(A)
    reduce(+, [sum(A[INDICES[k]:INDICES[k+1]-1]) for k in 1:last(size(A))])
end

foo(A)
bar(A)
baz(A)

@time for i in 1:1000 foo(A) end
@time for i in 1:1000 bar(A) end
@time for i in 1:1000 baz(A) end
