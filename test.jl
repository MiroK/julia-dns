# Test correctness

# First capture python's answer 
origstdout = STDOUT
(outread, outwrite) = redirect_stdout()
tic()
run(`python dns.py`)
py_time = toq()
py = parse(Float64, strip(readline(outread)))
redirect_stdout(origstdout)   # Cleanup

# julia's answer
include("dns.jl")
dns(2)
tic()
jl = dns(2^6)
jl_time = toq()

error = abs(py-jl)
println("Python time $(py_time), Julia time w/out JIT $(jl_time), Error $(error)")
