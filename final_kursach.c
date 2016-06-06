#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include <omp.h>

#define X_LOWER_BOUND 0.0
#define X_UPPER_BOUND 1.0

#define T_LOWER_BOUND 0.0
#define T_UPPER_BOUND 1.0

#define X_GRID_SIZE 20
#define T_GRID_SIZE 800

#define H (X_UPPER_BOUND - X_LOWER_BOUND) / X_GRID_SIZE   
#define TAU (T_UPPER_BOUND - T_LOWER_BOUND) / T_GRID_SIZE

#define SIGMA TAU / ( H * H )

#define A 1.0// we decide what values it may be
#define B -1.0
#define C 1.0

/*
kursach for Parallel Computing by Vladyslav Voloshyn
DA-31 group
Variant #6
explicit method
*/

void create_matrix(double ***array, int rows, int cols); 
void free_matrix(double ***array, int rows); 
void print_matrix(double** matrix , int rows, int cols); 

// formulas
double exact_solution_function(double x, double t); 
double calculate_next_layer_point(double previous, double current, double next);

// approximations for derivatives
double first_difference(double previous, double next); 
double second_difference(double previous, double current, double next);

double** calculate_numerical_result(int rank, int comm_size); // mpi
double** calculate_exact_result(); // openmp

double** calculate_errors(double** numerical_result, double** exact_result); // openmp
double calculate_average_error(double** matrix); // openmp


int main(int argc, char **argv)
{
	int rank, comm_size;        

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (comm_size < 2) {
    	printf("This task can only be exrcuted in 2 or more processes\n");
    	MPI_Abort(MPI_COMM_WORLD, 0);// Terminates MPI execution environment
    }

    double starttime, endtime;
    double** numerical_result = calculate_numerical_result(rank, comm_size);

    if (rank == 0)
    {
    	starttime = MPI_Wtime();// begin counting time of execution

		double** exact_result = calculate_exact_result();
		double** errors = calculate_errors(numerical_result, exact_result);
		double avg = calculate_average_error(errors);

		printf("\nCOURSE WORK 3RD GRADE\n\n");

		printf("\nNumerical Results: \n\n\n");
		print_matrix(numerical_result, T_GRID_SIZE, X_GRID_SIZE);

		printf("\nExact Results: \n\n\n");
		print_matrix(exact_result, T_GRID_SIZE, X_GRID_SIZE);

		printf("\nErrors: \n\n\n");
		print_matrix(errors, T_GRID_SIZE, X_GRID_SIZE);

		printf("\nAverage Error: %lf\n\n", avg);

		free_matrix(&numerical_result, T_GRID_SIZE);
		free_matrix(&exact_result, T_GRID_SIZE);
		free_matrix(&errors, T_GRID_SIZE);

		endtime = MPI_Wtime();// end counting time of execution
		printf("That took %f seconds.\n", endtime - starttime);
    }

	MPI_Finalize();
	return 0;
}


double exact_solution_function(double x, double t) 
{
	// My function w(x,t)
	return pow(sqrt(-B/A) + C * exp((7*A*t/12) + (sqrt(A/12)*x)), 0.25); 
}

double calculate_next_layer_point(double previous, double current, double next)
{
	// My function that I resolve from main equation
	return current + TAU * (second_difference(previous, current, next) + A * current + B * pow(current, 0.5));
}


void create_matrix(double ***array, int rows, int cols) 
{
	*array = (double**)malloc(sizeof(double) * rows);
	for (int i = 0; i < rows; ++i)
		(*array)[i] = calloc(cols, sizeof(double));
}

void free_matrix(double ***array, int rows) 
{
	for (int i = 0; i < rows; ++i)
		free((*array)[i]);
	free(*array);
}

void print_matrix(double** matrix , int rows, int cols) 
{
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++)
            printf("%.4f ", matrix[i][j]);
        printf("\n\n");
        if (i == 10)
        	break;
    }
}

double first_difference(double previous, double next) // first derivative
{
	return (next - previous) / TAU;
}

double second_difference(double previous, double current, double next) // second derivative
{	
	return (previous - 2.0 * current + next) / (H * H);
}	

double** calculate_numerical_result(int rank, int comm_size)
{
	double** local_grid;
	double previous, next, x;
	int local_grid_width = X_GRID_SIZE / comm_size;

	if (rank == comm_size - 1)
		local_grid_width += (X_GRID_SIZE % comm_size);// if there are some leftovers from dividing into comm_size

    create_matrix(&local_grid, T_GRID_SIZE, local_grid_width);

	for (int i = 0; i < local_grid_width; ++i)
	{
		if (rank == comm_size - 1)
			x = X_UPPER_BOUND - H * (local_grid_width - 1) + H * i;
		else 
			x = X_LOWER_BOUND + rank * H * local_grid_width + H * i;
		local_grid[0][i] = exact_solution_function(x, T_LOWER_BOUND);// w(0|1/20|1/10|3/20...., 0)
	}

	for (int k = 0; k < T_GRID_SIZE; ++k)
	{
		if (rank == 0)
			local_grid[k][0] = exact_solution_function(X_LOWER_BOUND, T_LOWER_BOUND + k * TAU);// w(0, 0|1/800|1/400|3/800...)
		else if (rank == comm_size - 1)
			local_grid[k][local_grid_width-1] = exact_solution_function(X_UPPER_BOUND, T_LOWER_BOUND + k * TAU);// w(1, 0|1/800|1/400|3/800...)
	}

	for (int k = 0; k < T_GRID_SIZE - 1; ++k)
	{
		if (rank != comm_size - 1)
			MPI_Send(&(local_grid[k][local_grid_width - 1]), 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
		if (rank != 0)
			MPI_Send(&(local_grid[k][0]), 1, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD);

		if (rank != 0)
			MPI_Recv(&(previous), 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (rank != comm_size - 1)
			MPI_Recv(&(next), 1, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE	);

		if (rank != comm_size - 1)
		{
			local_grid[k+1][local_grid_width - 1] = calculate_next_layer_point(local_grid[k][local_grid_width-2], local_grid[k][local_grid_width-1], next);
		}

		for (int i = 1; i < local_grid_width - 1; ++i)
		{
			local_grid[k+1][i] = calculate_next_layer_point(local_grid[k][i-1], local_grid[k][i], local_grid[k][i+1]);
		}
		
		if (rank != 0)
		{
			local_grid[k+1][0] = calculate_next_layer_point(previous, local_grid[k][0], local_grid[k][1]);
		}
	}


	double** global_grid = local_grid;
	if (rank == 0)
		create_matrix(&global_grid, T_GRID_SIZE, X_GRID_SIZE);
	
	int* counts = malloc(sizeof(int) * comm_size);
	int* displacements = malloc(sizeof(int) * comm_size);

	counts[0] = X_GRID_SIZE / comm_size;
	displacements[0] = 0;
	for (int i = 1; i < comm_size - 1; ++i)
	{
		counts[i] = X_GRID_SIZE / comm_size;
		displacements[i] = displacements[i-1] + counts[i];
	}
	counts[comm_size - 1] = counts[comm_size - 2] + (X_GRID_SIZE % comm_size);
	displacements[comm_size -1] = displacements[comm_size - 2] + counts[comm_size - 2];

	for (int k = 0; k < T_GRID_SIZE; ++k) {
		// Gathers into specified locations from all processes in a group
		MPI_Gatherv(&(local_grid[k][0]), counts[rank], MPI_DOUBLE, &(global_grid[k][0]), counts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}

	free(counts);
	free(displacements);
	free_matrix(&local_grid, T_GRID_SIZE);

	return global_grid;
}

double** calculate_exact_result() 
{
	double** exact;
	create_matrix(&exact, T_GRID_SIZE, X_GRID_SIZE);

	#pragma omp parallel shared(exact)
	{
		#pragma omp for
			for (int k = 0; k < T_GRID_SIZE; ++k) {
				for (int i = 0; i < X_GRID_SIZE; ++i) {
					exact[k][i] = exact_solution_function(X_LOWER_BOUND + i * H, T_LOWER_BOUND + k * TAU);
				}
			}
	}

	return exact;
}

double** calculate_errors(double** numerical_result, double** exact_result) 
{
	double** errors;
	create_matrix(&errors ,T_GRID_SIZE, X_GRID_SIZE);

	#pragma omp parallel shared(errors)
	{
		#pragma omp for
			for (int k = 0; k < T_GRID_SIZE; ++k) {
				for (int i = 0; i < X_GRID_SIZE; ++i) {
					errors[k][i] = 100 * fabs(exact_result[k][i] - numerical_result[k][i]) / exact_result[k][i]; // relative error
				}
			}
	}
	
	return errors;
}

double calculate_average_error(double** matrix) 
{
	double sum, average;

	#pragma omp parallel for reduction (+ : sum)
		for (int k = 1; k < T_GRID_SIZE; ++k) {
			for (int i = 1; i < X_GRID_SIZE-1; ++i) {
				sum += matrix[k][i];
			}
		}
	
	average = sum / ((T_GRID_SIZE - 1) * (X_GRID_SIZE - 2));

	return average;
}