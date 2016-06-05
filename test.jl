# Test correctness
if "skip" ∉  ARGS 
    # First capture python's answer 
    origstdout = STDOUT
    (outread, outwrite) = redirect_stdout()
    tic()
    run(`python dns.py`)
    py_time = toq()
    py = parse(Float64, strip(readline(outread)))
    py_min = parse(Float64, strip(readline(outread)))
    py_max = parse(Float64, strip(readline(outread)))
    redirect_stdout(origstdout)   # Cleanup
else
    py_time, py, py_min, py_max = -1, -1, -1, -1
end

version = (length(ARGS) > 0) ? ARGS[1] : 0
# julia's answer
include("./solvers/dns_$(version).jl")
dns(2)
jl_data = @timed dns(2^7)
jl, jl_time, jl_mem, jl_gc = jl_data[1:end-1]
jl, jl_min, jl_max = jl

error = abs(py-jl)
println("Error $(error)")
println("Python time $(py_time), Julia time w/out JIT $(jl_time): $(round(py_time/jl_time, 2))")
println("Min Time per tstep: Python $(py_min), Julia $(jl_min): $(round(py_min/jl_min, 2))")
println("Max Time per tstep: Python $(py_max), Julia $(jl_max): $(round(py_max/jl_max, 2))")
println("GC time $(jl_gc) is $(100*round(jl_gc/jl_time, 2))\% of exec")
println("Memory $(jl_mem/10^9)GB")
