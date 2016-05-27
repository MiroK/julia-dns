include("utils.jl")
using Utils

nu = 0.000625
T = 0.1
dt = 0.01
N = 2^2
Nh = N÷2+1

# Real vectors
U    = Array[RArray(N, N, N) for i in 1:3]
curl = Array[RArray(N, N, N) for i in 1:3]
# Complex vectors
U_hat  = Array[RArray(Nh, N, N) for i in 1:3]
U_hat0 = Array[RArray(Nh, N, N) for i in 1:3]
U_hat1 = Array[RArray(Nh, N, N) for i in 1:3]
dU     = Array[RArray(Nh, N, N) for i in 1:3]
# Complex scalars
Uc_hat = CArray(Nh, N, N)
P_hat  = CArray(Nh, N, N)
# Transpose
Uc_hatT = CArray(Nh, N, N)
# Real scalars
P = RArray(N, N, N)
# Real grid
x = collect(0:N-1)*2*pi/N
X = Array[ndgrid(x, x, x)...]
# Complex grid
kx = fftfreq(N, 1./N)
kz = kx[1:(N÷2+1)]; kz[end] *= -1
K = Array[ndgrid(kz, kx, kx)...]
# Square of wave number vectors
K2 = K[1].^2 + K[2].^2 + K[3].^2
# K/K2 term
K_over_K2 = Array[K[i]./K2 for i in 1:3]
# Fix division by zero
for i in 1:3 K_over_K2[i][1] = 0 end
# Dealising mask
kmax_dealias = 2*Nh/3
dealias =\
  (abs(K[1]).<kmax_dealias).*(abs(K[2]).<kmax_dealias).*(abs(K[3]).<kmax_dealias)
a = [1./6., 1./3., 1./3., 1./6.]  # Same!
b = [0.5, 0.5, 1.]                # Same!
