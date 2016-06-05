using MPI

np = parse(Int, first(ARGS))
manager = MPIManager(np=np)
addprocs(manager)

println("Added procs $(procs())")
@mpi_do manager (include("dns_2-impl.jl"); dns(2^2))
@mpi_do manager (dns(2^6))

exit()
