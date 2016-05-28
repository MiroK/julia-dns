# Test the utils functions against their python counterparts.
# Note that this requires PyCall and working python withon numpy and stuff

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

# Check FFTs
A = rand(5, 3, 7)
B = rfft(A, (1, 2, 3))   # This is done to get datatype of output
fA = zeros(B)
fftn_mpi!(A, fA)         # Fill fA
AA = zeros(A)            # Container for ifft of fA
ifftn_mpi!(fA, AA)       # Fill AA
@test maximum(abs(AA - A)) < 1E-13

println("Good to go!")
