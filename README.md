## Implementation of 3periodic spectral Navier-Stokes solver in Julia

### Background
Recently a pure Python based spectral DNS [solver](https://github.com/spectralDNS/spectralDNS) was preseted in a 
[paper](http://arxiv.org/pdf/1602.03638v1.pdf) by Mikael Mortesen and Hans Petter Langtangen. Relying only on 
vectorization and numpy, the solver runs slightly slower than the reference C implementation. Sprinkling
Cython in a few places then makes the solver achieve C speed. While performant the resulting code is 
very compact and easy to read/maintain/modify. In this project we explore if it is possible to obtain a similar
solver in Julia. By similar we mean code (i) that runs at C speed and (ii) is as compact as Python. 

### Organization
+ Initially we want to match/beat Python on a single CPU.
+ There will be multiple versions of solver `dns_i.jl`. Ideally, each one is faster than the predecessor and the ultimate one is at least on par with Python.
+ __This is very much a learning project!__

### Status
+ `dns_0.jl` is an implementation which strives to mimic Python as much as
  possible. Most noticably, we want U[i] to represent a slice of a 4d array. To
  this end 4d arrays are implemented as array of 3d arrrays. On average, Julia
  code is 1.6 times faster but `@time` reports upwards of 6GB allocated memory.
  So we are happy with speed but not with memory consumption.
+ `dns_1.jl`is not an evolution of `dns_0.jl`. It uses planned FFTs, proper 4d
  arrays, in-place operations (`axpy!`, `broadcast!`) and where in-place does
  not work, e.g. `.*=`, the loops are explicit. However we use linear indexing so
there are never 4 nested for loops. *This code runs 3.4 times faster than Python
  making it on par with C. More importantly it practically uses only the memory
  allocated for the data/work arrays* Compared to `dns_0.jl` the code did not
loose much of compactness. In summary our goal on a single CPU was achieved.
