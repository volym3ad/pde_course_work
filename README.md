# pde_course_work
forked and modified

To compile mpi & omp, run:

mpicc -o final final_kursach.c -std=c99 -lm -fopenmp

To run mpi & omp:

mpirun -np 4 ./final
