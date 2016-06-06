# pde_course_work
forked and modified

##explicit method

To compile mpi & omp, run:
```
mpicc -o final final_kursach.c -std=c99 -lm -fopenmp
```

To run mpi & omp:
```
mpirun -np 4 ./final
```
