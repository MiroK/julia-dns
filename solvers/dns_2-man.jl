using MPI

np = parse(Int, first(ARGS))
manager = MPIManager(np=np)
addprocs(manager)

println("Added procs $(procs())")
println("Running on $(np) processes")
@mpi_do manager (include("dns_2-impl.jl"); dns(2^6))
