define READ_ME

To compile:

	1) MPI's version, run in terminal : make mpi

	2) OMP's version, run in terminal : make omp

	3) (MPI + OMP)'s version, run in terminal : make mpi_omp

Or to clean everything: make clean

endef

export READ_ME
all:
	@echo "$$READ_ME"



mpi:
	(cd Mpi_version; make)
omp:
		(cd Omp_version; make)
mpi_omp:
		(cd Mpi_omp_version; make)


clean:
	(cd Mpi_version; make clean)

	(cd Omp_version; make clean)
	
	(cd Mpi_omp_version; make clean)
	
