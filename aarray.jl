type AArray
    list::Array

    function AArray(values::AbstractArray...)
        arrays = []
        for (i, v) in enumerate(values)
            push!(arrays, v)
        end
        new(arrays)
    end
end

#function getitem(array::Aarray, i)
#    @assert 1 <= length(array.array
#end

#b = AArray(collect(1:2), collect(3:4), collect(4:5))
#println(b[1]
