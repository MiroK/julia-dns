function foo()
    a = rand(100, 100)

    function bar()
        a[:] += 1
        sum(a)
    end
    
    bar()
end


function foo1()
    a = rand(100, 100)

    function bar1(a)
        a[:] += 1
        sum(a)
    end
    
    bar1(a)
end
