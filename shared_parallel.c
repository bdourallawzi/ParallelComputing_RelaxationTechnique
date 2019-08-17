#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#include "shared_parallel.h"

// var struct holding the command line input arguments as well as the global variables used cross-thread. 
struct{
    int array_dimension;
    int no_threads;
    double precision;
    double time_spent;
    int precision_achieved;
}var;

pthread_barrier_t sync_barrier;

/* relaxation_technique is called per pthread, with determined data per thread. It averages the four neighbouring values of a cell,
 after which the precision is checked. If the precision is reached for all the cells in this specific thread, then 
 the precision variable in the data struct parameter is set to 1. Once all the threads have reached the barrier, the main thread checks
 all the threads, if the precision variable in the data struct is set to 1. If so then precision is reached across the entire array and 
 the threads exit. Otherwise, the variable are re-set and the array pointers are swapped to started another iteration of the relaxation 
 technique across all threads.
 */
void *relaxation_technique(void *param_ptr) {
    data_threads *data = (data_threads *)param_ptr;

    printf("Thread %d, starting at (%d,%d) for %d cells.\n", 
        data->id, data->position_i, data->position_j, data->number_of_cells);

    int i, j, precision_count;
    int c = data -> number_of_cells;

    while(1){
        
        i = data -> position_i;
        j = data -> position_j;
        precision_count = 0;

        for (int count=0; count < c; count++){
            
            data -> avergaing_RW_ptr[i][j] = calc_avg(data->data_RO_ptr, i, j);

            if(precision_check(data -> data_RO_ptr, 
                data -> avergaing_RW_ptr, i, j) == 1){
                precision_count +=1;
            }

            j++;
            if (j > (var.array_dimension - 2)){
                i++; j = 1;
            }
        }
        if (precision_count == c){
            data -> precision = 1;
        }

        pthread_barrier_wait(&sync_barrier);
        pthread_barrier_wait(&sync_barrier);

        if(var.precision_achieved == 1){
            break;
        }

        // re-setting variables
        double **temp = data -> data_RO_ptr;
	    data -> data_RO_ptr = data -> avergaing_RW_ptr;
	    data -> avergaing_RW_ptr = temp;
        data -> precision = 0;

    }
    pthread_exit(NULL);
}

/* cross_thread_precision checks if the precision has been reached for all threads. The precision variable is set during 
relaxation technique calculation reducing the complexity of going through each cell of the array again. 
*/
int cross_thread_precision(data_threads *data_per_thread){
    for (int i = 0; i < var.no_threads; i++){
        data_threads *data = (data_threads *)&data_per_thread[i];
        if(data -> precision == 0){
            return 0;
        }
    }
    return 1;
}

/* calc_avg calcuates the average of 4 neighbouring values of a cell at position (i,j). 
The array parameter is read only to avoid race conditions.
*/
double calc_avg(double **a_data_RO, int i, int j){
    return (a_data_RO[i-1][j]+a_data_RO[i+1][j]+a_data_RO[i][j-1]+a_data_RO[i][j+1])/4;
}

/* precision_check checks the precision between the previous value and the newly calculated value. 
This is done per cell after the averaging calculation.
*/
int precision_check(double **a, double **a_data, int i, int j){
    if (fabs(a[i][j]-a_data[i][j]) > var.precision){
        return 0;
    }
    return 1;
}

/* the main function initialises all the variables.
After the array and data per thread is determines. 
N number of pthreads are created calling relaxation_technique with the data corresponding to the thread refernece. 
A barrier for the number of threads + the main function is created, it is in place to ensure synchronisation of
the average calculation and to check the precision in the main thread after each iteration of the entire array. 
If the precision is not reached, the barrier and the pthreads are re-used completing another iteration of the
relaxation technique.  
*/
int main (int argc, char *argv[]){
    set_arg_variables(argc, argv);
    int count = 0;
    
    double **a = create_2d_array();
    initialise_random_square_matrix(a);
    double **a_copy = duplicate_2d_array(a);
    //printf("Initial %dx%d array\n", var.array_dimension, var.array_dimension);
    //print_2d_array(a); 


    data_threads *data_per_thread = malloc(var.no_threads*sizeof(data_threads)); 
    if (data_per_thread == NULL){
        printf("Could not allocate memory for thread data.\n");
    }
    initialise_data(data_per_thread, a, a_copy);

    pthread_t *threads = malloc(var.no_threads*sizeof(data_threads));
    if (threads == NULL){
        printf("Could not allocate memory for threads.");
    }
    pthread_barrier_init(&sync_barrier, NULL, var.no_threads+1);

    for (int i=0; i < var.no_threads; i++){
        if (pthread_create(&threads[i], NULL, relaxation_technique, 
            &data_per_thread[i])){
            fprintf(stderr, "Error creating thread %d\n", i);
        }
    }

    while (1){
        count++;
        
        pthread_barrier_wait(&sync_barrier);

        if(cross_thread_precision(data_per_thread)==1){
            printf("Precision across entire array reached in %d iterations!\n", 
            count);
            var.precision_achieved = 1;
            pthread_barrier_wait(&sync_barrier); // sync and exit
            break;
        }

        printf("Iteration %d complete. Precision still not reached across the entire array. Undergoing another iteration.\n", count);
        pthread_barrier_wait(&sync_barrier); // sync and loop
    }

    for (int i = 0; i < var.no_threads; i++){
        if( pthread_join(threads[i], NULL)){
            fprintf(stderr, "Error joining thread\n");
        }
    }


    //printf("Relaxation technique result.\n");
    //print_2d_array(a_copy);

    free_2d_array(a);
    free_2d_array(a_copy);
    free(data_per_thread);
    free(threads);  
}

/* initialise_data sets the data per thread in a struct containing all the values needed per thread. 
Each thread is given an id, a certain number of cells to work on with a starting position.  
This ensures that there is no overall lap between threads working on the same data that could result in 
race conditions. Pointers to the array and a duplicate of it is intially set. The 2 pointers become 
one that is readonly taking the data to averaged and then updated to the other write array. 
*/
void initialise_data(data_threads *data_per_thread, double **a, double **a_copy){
    int inner_cells = (var.array_dimension-2)*(var.array_dimension-2);
    
    if (inner_cells % var.no_threads > 0){
        printf("Work not divided equally to all threads.\n");
        int remainder = inner_cells % var.no_threads;
        int inner_per_thread = inner_cells / var.no_threads;
        int i = 0, j = 0, next = 0;
        for (int t=0; t < var.no_threads; t++){
            data_per_thread[t].number_of_cells = inner_per_thread;
            if (remainder > 0){
                data_per_thread[t].number_of_cells += 1;
                remainder --; 
            }
            data_per_thread[t].id = t;
            data_per_thread[t].data_RO_ptr = a;
            data_per_thread[t].avergaing_RW_ptr = a_copy;
            data_per_thread[t].position_i = i+1;
            data_per_thread[t].position_j = j+1;
            data_per_thread[t].precision = 0;
            next += data_per_thread[t].number_of_cells;
            i = next / (var.array_dimension-2);
            j = next % (var.array_dimension-2);
        }            
    }else{
        printf("Work divided equally to all threads.\n");
        int inner_per_thread = inner_cells / var.no_threads;
        int i = 0, j = 0, next = 0;
        for (int t = 0; t < var.no_threads; t++){
            data_per_thread[t].id = t;
            data_per_thread[t].number_of_cells = inner_per_thread;
            data_per_thread[t].data_RO_ptr = a;
            data_per_thread[t].avergaing_RW_ptr = a_copy;
            data_per_thread[t].position_i = i+1;
            data_per_thread[t].position_j = j+1;
            data_per_thread[t].precision = 0;
            next += data_per_thread[t].number_of_cells;
            i = next / (var.array_dimension-2);
            j = next % (var.array_dimension-2);
        } 
    }
}

/* set_arg_variables looks at the command line arguments and sets the variables needed.
*/
void set_arg_variables(int argc, char *argv[]){
    
    if (argc == 4){
        printf("Using the following arguments passed:\n");
        printf("Array dimension:%s, Number of threads: %s, Precision to be reached: %s\n", 
        argv[1], argv[2], argv[3]);
        var.array_dimension = atoi(argv[1]);
        var.no_threads = atoi(argv[2]);
        var.precision = atof(argv[3]);
    }else{
        printf("Expecting 5 arguments; program name, square array dimension, number of threads, precision and file name to save data. Arguments not set correclty.\n");
        printf("Default values used: array dimension 10, ");
        var.array_dimension = 10;
        var.no_threads = 2;
        var.precision = 0.1;
    }
    var.precision_achieved = 0;
    var.time_spent = 0.0;
}

/* initialise_random_square_matrix gives a constant random value to an array of n dimensions 
set by the command line arguments with a constant edge value.
*/
void initialise_random_square_matrix(double **a){
     for (int i = 0; i < var.array_dimension; i++){
        for (int j =0; j < var.array_dimension; j++){
            if (i == 0 || j == 0 || i == (var.array_dimension-1) || 
            j == (var.array_dimension-1)){
				a[i][j] = (double)1; 
			}else{
				a[i][j] = ((double)rand()/(double)RAND_MAX);
            }
        }
    }
}

/* a 2D array is just a list of pointers to 1D arrays. 
A function to create an array is a way to avoid repeatative and fidely code.
Also helps with memory leaks as easier to match each create with a free. 
*/
double** create_2d_array(){
    double** a = malloc(var.array_dimension * sizeof(double*));
    if (a == NULL) {
        return NULL;
    }

    int i;
    for (i = 0; i < var.array_dimension; i++) {
        a[i] = malloc(var.array_dimension * sizeof(double));

        if (a[i] == NULL) {
            free(a);

            return NULL;
        }
    }

    return a;
}

/* duplicates an array that is passed as param. 
*/
double** duplicate_2d_array(double** a){
    double** a_copy = create_2d_array(var.array_dimension);

    for (int i = 0; i < var.array_dimension; i++) {
        a_copy[i] = memcpy(a_copy[i], a[i], var.array_dimension * sizeof(double));
    }

    return a_copy;
}

/* free_2d_array is a function passing in a 2D array. The following helps avoid repeatative and fidely code.
Also helps with memory leaks as easier to match each create with a free. 
*/
void free_2d_array(double **a){
    int i;
    for (i = 0; i < var.array_dimension; i++) {
        free(a[i]);
    }

    free(a);
}

/* prints a 2d array
*/
void print_2d_array(double** a){
    int i, j;
    for (i = 0; i < var.array_dimension; i++) {
        for (j = 0; j < var.array_dimension; j++) {
            printf("%10f ", a[i][j]);
        }

        printf("\n");
    }

    printf("\n");
}