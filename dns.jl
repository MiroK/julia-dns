include("utils.jl")
using Utils

nu = 0.000625
T = 0.1
dt = 0.01
N = 2^2

num_processes = 1
rank = 0
Np = N÷num_processes
Nh = N÷2+1

typealias RealT Float64
typealias CmplT Complex64
typealias RArray Array{RealT}
typealias CArray Array{CmplT}



P_hat = CArray(Nh, Np, N)
# Transpose
Uc_hatT = CArray(Nh, N, Np)
# Real scalars
P = RArray(N, N, Np)
# MPI
U_mpi = [CArray(Nh, Np, Np) for i in 1:num_processes]
# Real grid
x = collect(0:N-1)*2*pi/N
X = Array[ndgrid(x, x, collect(rank*Np:(rank+1)*Np-1)*2*pi-N)...]


# 
# 
# kx = fftfreq(N, 1./N)
# kz = kx[1:(N÷2+1)]; kz[end] *= -1
# K = ndgrid(kx, kx[(rank*Np+1):((rank+1)*Np-1)], kz)
# K2 = sum(K*K, 0, dtype=int)
# K_over_K2 = K.astype(float) / where(K2 == 0, 1, K2).astype(float)
# kmax_dealias = 2./3.*(N/2+1)
# dealias = array((abs(K[0]) < kmax_dealias)*(abs(K[1]) < kmax_dealias)*
#                 (abs(K[2]) < kmax_dealias), dtype=bool)
a = [1./6., 1./3., 1./3., 1./6.]  # Same!
b = [0.5, 0.5, 1.]                # Same!
