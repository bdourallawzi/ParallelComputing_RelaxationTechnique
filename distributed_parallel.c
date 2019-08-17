#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

/*
Compiling

    mpicc -Wall -std=gnu99 bal28_distributed.c -o bal28_distributed

Execution

    mpirun -np 3 mpi distributed_parallel 10 0.01 sample.csv 

*/

struct ScatterParam{
    /* integer array (of length group num_processes) specifying the number of elements to send to each processor */
    int* send_counts_a;
    /* integer array (of length group num_processes). Entry i specifies the displacement
    - relative to the send buffer (the array) from which to take the outgoing data to process i*/
    int* scatter_displs;
    /* number of elements in the recieve buffer */
    int recieve_count;
};

struct GatherParam{
    /* integer number of elements in send buffer */
    int send_count;
    /* integer array (of length group num_processes) containing the number of elements that are recieved from each process 
    significant at root */
    int* recieve_counts_a;
    /* integer array (of length group num_processes). Entry i specifies the displacement 
    - relative to the recieve buffer at which to place the incoming data from process i
    significant at root */
    int* recieve_displs;
};

void set_arg_variables(int, char**);
void initialise_array(double*, int);
void relax_array(double*, double*, struct ScatterParam*, struct GatherParam*, double*, double*);
void initialise_data(struct ScatterParam*, struct GatherParam*, int, int);
void swap(double**, double**);
double calc_average(double, double, double, double);
int precision_check(double, double);
void print_array(double*);
void record_variables(char*, double, int);

struct{
    int array_dimension;
    double precision;
    char* file_name;
} var;

int main (int argc, char **argv){
    double time = 0.0;

    // Initialise the MPI environment
    int ierr, num_processes, rank;
    ierr = MPI_Init(&argc, &argv);
    // Get the number of processes
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    // Get the rank of the process
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    MPI_Barrier(MPI_COMM_WORLD);
    time -= MPI_Wtime();

    // Set the environment variables passed at runtime 
	set_arg_variables(argc, argv);
    double *a = malloc(var.array_dimension*var.array_dimension*sizeof(double));
    if (a == NULL){ 
		printf("Couldn't allocate memory");
	}

    // Initialise a square array 
    initialise_array(a, var.array_dimension);

    /* define data structure to hold the array values to be computed for process. */
    struct ScatterParam scatter_param;
    struct GatherParam gather_param;
    /* integer array where each cell corresponding to a process 
     holding the displacement of the data needed for scatterv(as it can be irregular)*/
    scatter_param.scatter_displs = malloc((num_processes)*sizeof(int));
    /* integer array where each cell corresponding to a process 
    holding the number of cells needed from the buffer (the array a in this case)*/
	scatter_param.send_counts_a = malloc((num_processes)*sizeof(int));
    /* integer array where each cell coresponding to a process
    holding the displacement of the data to be gathered from the recieve buffer*/
	gather_param.recieve_displs =  malloc((num_processes)*sizeof(int));
    /* integer array where each cell corresponding to a process
    holding the number of cells to be gathered from the recieve buffer*/
    gather_param.recieve_counts_a =  malloc((num_processes)*sizeof(int));

    /* calls function initialise data to determine the data needed per process. */
    initialise_data(&scatter_param, &gather_param, rank, num_processes);

    /* after the initialise_data function, we have the data parameters needed to complete
    the MPI scatter and gather. 
    the recieve_count determined in the initialise_data function is the number of elements in the recvbuf
    the send_count determined in the initialise_data function is the number of elements in the sendbuf
    more detail on the calculation throughout the initialise_data function*/
    double *recvbuf = malloc(scatter_param.recieve_count * sizeof(double));
	double *sendbuf = malloc(gather_param.send_count * sizeof(double));    
    /* readonly array that will be used to gather the newly averaged values 
    and then swapped with the other array in order to avoid data race conditions. */
    double *a_data = malloc(var.array_dimension*var.array_dimension*sizeof(double));
    if (a_data == NULL){ 
		printf("Couldn't allocate memory");
	}
    initialise_array(a_data, var.array_dimension);

    /* relax_array is called along with the data needed for completing the relaxation technique.
    */
    relax_array(a, a_data, &scatter_param, &gather_param, recvbuf, sendbuf);

    MPI_Barrier(MPI_COMM_WORLD);
    time += MPI_Wtime();    

    MPI_Finalize();
    
    if (rank == 0){
        record_variables(var.file_name, time, num_processes);
    }

    free(recvbuf);
    free(sendbuf);
	free(a);
    free(a_data);
    free(scatter_param.send_counts_a);
    free(scatter_param.scatter_displs);
    free(gather_param.recieve_counts_a);
    free(gather_param.recieve_displs);
    return ierr ? 1 : 0;

}

/* relax_array function averages the four neighbouring values of a cell, checking the precision at each cell. 
*/
void relax_array(double *a, double *a_data, struct ScatterParam *scatter_param, struct GatherParam *gather_param, double *recvbuf, double* sendbuf){
    int precision_per_process = 1;
    int global_precision = 0;

	// as the first row is fixed, the relaxation would start from the second row of the matrix
	int start = var.array_dimension;
	// as the last row is fixed, the relaxation would stop a row before the end of the matrix
	int end = scatter_param->recieve_count - var.array_dimension - 1;
	// array counter, used to store values  
	int count = 0;
    int iterations=1;

    while(!global_precision){
        /* scatter a buffer in parts to all processes */
	    MPI_Scatterv(a, scatter_param->send_counts_a, scatter_param->scatter_displs, MPI_DOUBLE, recvbuf, 
   				    scatter_param->recieve_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        count = 0;
        /* precision is checked across each process. The values are then all reduced to check is precision is reached
        across all */
        precision_per_process = 1;
        for (int i = start; i <= end; i++){
            // if statement need to check if it is an edge or not 
            if ((i % var.array_dimension != 0) && ((i+1) % var.array_dimension != 0)){
                // newly averaged array written into the send buffer
                sendbuf[count] = calc_average(	recvbuf[i-1], recvbuf[i+1],
                                                recvbuf[i+var.array_dimension],
                                                recvbuf[i-var.array_dimension]);
                // checks the precision of the newly calculated value for each computation
                int p = precision_check(sendbuf[count], recvbuf[i]);
                precision_per_process = precision_per_process && p;
            }else{
                // edge values are fixed
                sendbuf[count] = recvbuf[i];
            }
            count++;
	    }
       
        /* combines of value precision per process from all processes, computes a logical and across.
        The output parameter then holds if the precision is reached across the entire matrix (1), otherwise
        (0).  
        If precision is reached the loop will break and the relaxation technique is complete. */
        MPI_Allreduce(&precision_per_process, &global_precision, 1, 
                    MPI_INT, MPI_LAND, MPI_COMM_WORLD);

        /* gathers into specified locations based on the displacement array initsalised from all processes, 
        retrieving the newly averaged values */
        MPI_Allgatherv(sendbuf, gather_param->send_count, MPI_DOUBLE, a_data, gather_param->recieve_counts_a,
    			gather_param->recieve_displs, MPI_DOUBLE, MPI_COMM_WORLD);

        swap(&a, &a_data);
        iterations++;
    }
}

/* initialise_data function determines the data needed for the MPI function calls.
MPI processes are used to complete the computation in parallel over distributed memory. 
*/
void initialise_data(struct ScatterParam *scatter_param, struct GatherParam *gather_param, int rank, int num_processes){
    scatter_param->scatter_displs = malloc((num_processes)*sizeof(int));
	scatter_param->send_counts_a = malloc((num_processes)*sizeof(int));
	gather_param->recieve_displs =  malloc((num_processes)*sizeof(int));
    gather_param->recieve_counts_a =  malloc((num_processes)*sizeof(int));

    /* number of rows that need to be averaged. 
    To be appropriately, divided to each process */
    int inner_rows = var.array_dimension-2;
	/* average number of rows, to be given to each process*/
	int avg_row_per_process = inner_rows / num_processes;
    /* the remainder of rows to be distribued to the processes*/
    int remainder =  inner_rows % num_processes;

	for (int i = 0; i < num_processes; i++){
        /*scatter_displs is an integer array whereby each cell of the displs array 
        would hold the starting position to take the data from the sendbuf to the 
        be procssed corresponding to the rank. 
        The displacement has to start at the cell needed for the averaging, not solely the
        row that is going to be averaged. */
		scatter_param->scatter_displs[i] = i * avg_row_per_process * var.array_dimension;
		/* recieve_displs is an integer array whereby each cell of the displs array holds 
        the starting the position of which it gather elements from the buffer.
        The cell corresponding to the process rank. */
		gather_param->recieve_displs[i] = var.array_dimension +
									(i * avg_row_per_process * var.array_dimension);

		/* the condition is placed in the case where the work is not constant 
        remainder != 0, so that last process would process the remaining rows*/
		if (i < num_processes-1){
			/* send_counts_a is an integer array specifiying the number of elements 
            to send to each processor from the sendbuf(which is the array in this case). 
            The calculation is based on the row/s of the matrix it will be processing
            as well as the row above and below it such that averaging can be completed. */
			scatter_param->send_counts_a[i] = (avg_row_per_process + 2) * var.array_dimension;
			/* recieve_counts_a is an integer array specifiying the number of elements 
            to be gathered in the recieve buffer corresponding to the rank/process */
			gather_param->recieve_counts_a[i] = avg_row_per_process * var.array_dimension;
		}else{
            /* as not each process will not necessarily recieve the same number of rows/elements,
            the last process will be calculating the remaning rows. */
			scatter_param->send_counts_a[i] = (avg_row_per_process + remainder + 2)
                                                * var.array_dimension;
			gather_param->recieve_counts_a[i] = (avg_row_per_process + remainder) 
                                                *var.array_dimension;
		}
        
	}

    /* if it is the last process, which could possibly be unequal to the other processes -
        hold the remainder*/
	if (rank == (num_processes-1)){

		/* recieve_count is the total number of elements to be returned -  the
        total number of cells computed (only the rows that have been averaged 
        - don't need the rows above and below it)
        In the else case, all will be the same size, the last process will otherwise
        may have an unequal number that has been calculated.*/
		scatter_param->recieve_count = (avg_row_per_process + 
			((var.array_dimension-2) % num_processes) + 2) * var.array_dimension;
        /* send_count is the total number of elements in the send buffer
        */
		gather_param->send_count = (avg_row_per_process + 
			((var.array_dimension-2) % num_processes)) * var.array_dimension;
			
	}else{
		scatter_param->recieve_count = (avg_row_per_process + 2) * var.array_dimension;
		gather_param->send_count = avg_row_per_process * var.array_dimension;
	}

}

/* swap the pointers of the arrays one acting as a read only while the other is updated and averaged. 
This is done to avoid race conditions.*/
void swap(double **a, double **a_data){
	//printf("SWAP\n");
    double *temp = *a;
	*a = *a_data;
	*a_data = temp;
}

/* precision_check checks the precision between the previous value and the newly calculated value. 
This is done per cell after the averaging calculation.
*/
int precision_check(double v_averaged, double v){
	if(fabs(v_averaged-v) > var.precision){
		return 0;
	}
	return 1;
}

/* calc_avg calcuates the average of 4 neighbouring values of the parameters passed. 
*/
double calc_average(double a, double b, double c, double d){
	return (a + b + c + d)/4;
}

/* initialise_array gives a constant random value to an array of n dimensions 
set by the command line arguments with a constant edge value.
*/
void initialise_array(double *a, int dimension){
	for (int i = 0; i < (dimension*dimension); i++){
		if(i < dimension || i % dimension == 0 || (i+1) % dimension == 0 ||
		((i >= dimension * dimension - dimension) && (i < dimension * dimension))){
			a[i] = 1.0;
		}else{
			a[i] = 0.0;
		}
	}
}


/* prints a 2d array
*/
void print_array(double *a){
	
	for (int i = 0; i < (var.array_dimension * var.array_dimension); i++){
		if (i % var.array_dimension == 0){
			printf("\n");
		}
		printf("%f ",a[i]);
	}
    printf("\n");
}

/* set_arg_variables looks at the command line arguments and sets the variables needed.
*/
void set_arg_variables(int argc, char **argv){
    if (argc == 4){
        var.array_dimension = atoi(argv[1]);
        var.precision = atof(argv[2]);
        var.file_name = argv[3];
    }else{
        var.array_dimension = 10;
        var.precision = 0.01;
        var.file_name = "dis_parallel.csv";
    }
}

/* record_variables was used to collect data for scalability investigation running the program on balena
*/
void record_variables(char* filename, double time_taken, int num_processes){
    FILE *fp;

    if (access(filename, F_OK) != -1){
        fp = fopen(filename, "a");
        fprintf(fp, "-----------------------------------------------------------------\n");
    }else{
        fp = fopen(filename, "w");
        fprintf(fp,"Number of processes, Array Dimension, Precision, Time Taken\n");
    }

    fprintf(fp, "%d, %d, %f, %f\n", num_processes, var.array_dimension, 
    var.precision, time_taken);


    printf("num processes: %d, dimension: %d, precision: %f, time taken: %f\n", 
    num_processes, var.array_dimension, var.precision, time_taken);

    fclose(fp);
    printf("%s file created\n", filename);
}