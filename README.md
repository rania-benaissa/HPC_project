# HPC_project

Parallel vector research in the kernel of large sparse matrices modulo
p using the block-Lanczos algorithm.


## Context

The project's aim is to parallelize a sequential program (which is provided) that performs the resolution of linear systems of type : $xM = 0$ mod p, where M is a sparse matrix of size $N  * (N - k)$. You can also refer to my [ project report (in french)](projet_HPC_BENAISSA_SHAO.pdf)

There are three parallelized version:

1. Using MPI
2. Using OpenMP
3. Using both (MPI + OpenMP)

## Usage

To compile each version:

1. MPI's version, run in terminal : ```make mpi```

2. OMP's version, run in terminal : ```make omp```

3. (MPI + OMP)'s version, run in terminal : ```make mpi_omp```

Or to clean the project : ```make clean```



