## Implementation of 3periodic spectral Navier-Stokes solver in Julia

### Background
Recently a pure python based spectral DNS [solver](https://github.com/spectralDNS/spectralDNS) was preseted in a 
[paper](http://arxiv.org/pdf/1602.03638v1.pdf) by Mikael Mortesen and Hans Petter Langtangen. Relying only on 
vectorization and numpy, the solver runs slightly slower than the reference C implementation. Sprinkling
Cython in a few places then makes the solver achieve C speed. While performant the resulting code is 
very compact and easy to read/maintain/modify. In this project we explore if it is possible to obtain a similar
solver in Julia. By similar we mean code (i) that runs at C speed and (ii) is as compact as Python. 



