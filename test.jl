include("utils.jl")
using Utils
using Base.Test

# Check if the functions do same as in python
using PyCall
@pyimport numpy as np

# NDGRID
# NOTE np.mgrid[] does not work because julia sees it as indexing
py_grid = np.meshgrid(collect(1:3), collect(4:5), indexing="ij")
jl_grid = Utils.ndgrid(1:3, 4:5)
@test_approx_eq_eps norm(first(py_grid) - first(jl_grid)) 0 1E-13
@test_approx_eq_eps norm(last(py_grid) - last(jl_grid)) 0 1E-13

@pyimport numpy.fft as npfft
# FFTFREQ
py_freq = npfft.fftfreq(31, 0.123)
jl_freq = Utils.fftfreq(31, 0.123)
@test_approx_eq_eps norm(py_freq - jl_freq) 0 1E-13
py_freq = npfft.fftfreq(20, 0.42)
jl_freq = Utils.fftfreq(20, 0.42)
@test_approx_eq_eps norm(py_freq - jl_freq) 0 1E-13

# RANGE stuff
jl = collect(0:10-1)
py = np.arange(0, 10)
@test_approx_eq_eps norm(py-jl) 0 1E-13

# Wave numbers check
N = Np = 4
rank = 0
kx = fftfreq(N, 1./N)
kz = kx[1:(N÷2+1)]; kz[end] *= -1
jl_K = ndgrid(kx, kx[(rank*Np+1):((rank+1)*Np-1)], kz)
py_K = np.meshgrid(kx, kx[(rank*Np+1):((rank+1)*Np-1)], kz, indexing="ij")
# Are K correct?
@test_approx_eq_eps maximum([maximum(abs(jl-py)) for (jl, py) in zip(jl_K, py_K)]) 0 1E-13
# Are K**2 correct

# K over and dealiasing

println("Good to go!")
