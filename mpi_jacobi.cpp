/**
 * @file    mpi_jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements MPI functions for distributing vectors and matrixes,
 *          parallel distributed matrix-vector multiplication and Jacobi's
 *          method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#include "mpi_jacobi.h"
#include "jacobi.h"
#include "utils.h"

#include <stdlib.h>
#include <cstring>
#include <assert.h>
#include <math.h>
#include <vector>

/*
 * TODO: Implement your solutions here
 */


// distribute the local vector at processor (0,0) among (i,0)/first col
void distribute_vector(const int n, double* input_vector, double** local_vector, MPI_Comm comm)
{
    int rank,coords[2];
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 2, &coords[0]);

    //work only on the columns
    int keep_dims[2] = {1, 0};
    MPI_Comm comm_col;
    MPI_Cart_sub(comm, keep_dims, &comm_col);
    
    if (coords[1] == 0) {
        //initial parameter for MPI_Scatterv
        int size;
        MPI_Comm_size(comm, &size);
        int q = (int) sqrt(size);
        int local_size = block_decompose(n, q, coords[0]);
        int sendcounts[q],displs[q];
        *local_vector = (double*)malloc(sizeof(double)*local_size);
        for (int i = 0; i < q; i++) {
            sendcounts[i] = block_decompose(n, q, i);
            displs[i] = (i == 0) ? 0 : (displs[i - 1] + sendcounts[i - 1]);
        }
        MPI_Scatterv(input_vector, sendcounts, displs, MPI_DOUBLE, *local_vector, local_size, MPI_DOUBLE, 0, comm_col);
    }
    MPI_Comm_free(&comm_col);
    return;
}

// gather the local vector distributed among (i,0) to the processor (0,0)
void gather_vector(const int n, double* local_vector, double* output_vector, MPI_Comm comm)
{
    int rank,coords[2];
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 2, &coords[0]);

    //work only on the columns
    int keep_dims[2] = {true, false};
    MPI_Comm comm_col;
    MPI_Cart_sub(comm, keep_dims, &comm_col);

    if (coords[1] == 0) {
        //initial parameter for MPI_Gatherv
        int size;
        MPI_Comm_size(comm, &size);
        int q = (int) sqrt(size);
        int local_size = block_decompose(n, q, coords[0]);
        int recvcounts[q],displs[q];
        for (int i = 0; i < q; i++) {
            recvcounts[i] = block_decompose(n, q, i);
            if (i==0){displs[i]=0;}
            else{displs[i]=displs[i - 1] + recvcounts[i - 1];}
        }
        MPI_Gatherv(local_vector, local_size, MPI_DOUBLE, output_vector, recvcounts, displs, MPI_DOUBLE, 0, comm_col);
    }  
    MPI_Comm_free(&comm_col);
    return;
}
// for distributing matrix, we scatter the matrix along the first column so that each row has all its elements in its first column. Then we create a MPI data structure to scatter the data in the first column to its corresponding column along the row.
void distribute_matrix(const int n, double* input_matrix, double** local_matrix, MPI_Comm comm)
{
    // 2 phase
    int rank, size, coords[2];
    double *c1matrix;
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 2, &coords[0]);
    MPI_Comm_size(comm, &size);
    int q = (int) sqrt(size);
    
    // col sub topology
    int keep_dims_col[2] = {1, 0};
    MPI_Comm comm_col;
    MPI_Cart_sub(comm, keep_dims_col, &comm_col);

    // row sub topology
    int keep_dims_row[2] = {0, 1};
    MPI_Comm comm_row;
    MPI_Cart_sub(comm, keep_dims_row, &comm_row);

    int local_row_size = block_decompose(n, q, coords[0]);
    int local_col_size = block_decompose(n, q, coords[1]);

    if (coords[1] == 0) {
        c1matrix = (double*)malloc(sizeof(double)*local_row_size*n);
        int sendcounts[q], displs[q];
        for (int i = 0; i < q; i++) {
            sendcounts[i] = n * block_decompose(n, q, i);
            if (i==0){displs[i]=0;}
            else{displs[i]=displs[i - 1] + sendcounts[i - 1];}
        }

        // all n col in the first col
        MPI_Scatterv(input_matrix, sendcounts, displs, MPI_DOUBLE, c1matrix, n*local_row_size, MPI_DOUBLE,
                     0, comm_col);
    }

    MPI_Datatype vec, localvec;
    MPI_Type_vector(local_row_size, 1, n, MPI_DOUBLE, &vec);
    MPI_Type_create_resized(vec, 0, sizeof(double), &vec);
    MPI_Type_commit(&vec);

    MPI_Type_vector(local_row_size, 1, local_col_size, MPI_DOUBLE, &localvec);
    MPI_Type_create_resized(localvec, 0, sizeof(double), &localvec);
    MPI_Type_commit(&localvec);

    int sendcounts[q],senddispls[q];
    for (int col=0; col<q; col++) {
        sendcounts[col] = block_decompose(n, q, col);
        senddispls[col]=(col==0)? 0 : (senddispls[col - 1] + sendcounts[col - 1]);
    }

    double *rowptr = (coords[1] == 0) ? &c1matrix[0] : NULL;
    (*local_matrix) =  (double*)malloc(sizeof(double)*local_row_size*local_col_size); 

    MPI_Scatterv(rowptr, sendcounts, senddispls, vec,
                  *local_matrix, sendcounts[coords[1]], localvec, 0, comm_row);

    MPI_Type_free(&localvec);
    MPI_Type_free(&vec);
    if (coords[1] == 0) {
        free(c1matrix);
    }
    MPI_Comm_free(&comm_col);
    MPI_Comm_free(&comm_row);
    return;
}



void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm)
{
    int my2drank, mycoords[2], size;
    MPI_Comm_size(comm, &size);
    int q = (int) sqrt(size);
    MPI_Comm_rank(comm, &my2drank);
    MPI_Cart_coords(comm, my2drank, 2, &mycoords[0]);

    // col sub topology
    int keep_dims_col[2] = {1, 0};
    MPI_Comm comm_col;
    MPI_Cart_sub(comm, keep_dims_col, &comm_col);

    // row sub topology
    int keep_dims_row[2] = {0, 1};
    MPI_Comm comm_row;
    MPI_Cart_sub(comm, keep_dims_row, &comm_row);

    int local_row_size=block_decompose(n, q, mycoords[0]);
    int local_col_size=block_decompose(n, q, mycoords[1]);
    // root don't need to send.
    if(my2drank == 0) {
        std::memcpy (row_vector, col_vector, (n/q+1)*sizeof(double));
    }else if(mycoords[1]==0){
        MPI_Send(col_vector,local_row_size,MPI_DOUBLE, mycoords[0],1,comm_row);
    }else if(mycoords[0]==mycoords[1]){
        MPI_Recv(row_vector,local_row_size, MPI_DOUBLE,0,1,comm_row,MPI_STATUS_IGNORE);
    }
    MPI_Bcast(row_vector,local_col_size, MPI_DOUBLE, mycoords[1],comm_col);
 
    MPI_Comm_free(&comm_col);
    MPI_Comm_free(&comm_row);
    return;
}

// distributed multiplication across all the processor and gather the result on the first column
void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm)
{
    int my2drank, mycoords[2], size;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &my2drank);
    MPI_Cart_coords(comm, my2drank, 2, &mycoords[0]);
    int q = (int) sqrt(size);
    int local_row_size=block_decompose(n, q, mycoords[0]);
    int local_col_size=block_decompose(n, q, mycoords[1]);
    // distributed result
    double* local_tmp_y = new double[local_row_size];
    // distributed x
    double * local_tmp_x = new double[local_col_size];

    // transpose_bcast_vector local_x, which is initially on col1
    transpose_bcast_vector(n, local_x, local_tmp_x, comm);

    matrix_vector_mult(local_row_size, local_col_size, local_A, local_tmp_x, local_tmp_y);

    //create row base sub topology
    MPI_Comm comm_row;
    int keep_dims[2]={0,1};
    MPI_Cart_sub(comm,keep_dims,&comm_row);

    MPI_Reduce(local_tmp_y, local_y, local_row_size, MPI_DOUBLE, MPI_SUM, 0, comm_row);

    delete[] local_tmp_y;
    delete[] local_tmp_x;
    MPI_Comm_free(&comm_row);
    return;
}


// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
                MPI_Comm comm, int max_iter, double l2_termination)
{
    int rank, size, coordinate[2];
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 2, &coordinate[0]);
    MPI_Comm_size(comm, &size);
    int q = (int) sqrt(size);
    
    // col sub topology
    int keep_dims_col[2] = {1, 0};
    MPI_Comm comm_col;
    MPI_Cart_sub(comm, keep_dims_col, &comm_col);

    // row sub topology
    int keep_dims_row[2] = {0, 1};
    MPI_Comm comm_row;
    MPI_Cart_sub(comm, keep_dims_row, &comm_row);

    int local_row_size = block_decompose(n, q, coordinate[0]);
    int local_col_size = block_decompose(n, q, coordinate[1]);

    double *local_D = (double*)malloc(sizeof(double)*local_row_size);
    double *local_R = (double*)malloc(sizeof(double)*local_col_size*local_row_size);
    memset (local_x, 0, sizeof (double) * local_row_size);  
    std::memcpy(local_R, local_A, sizeof(double)*local_row_size*local_col_size);

    // initial R and D
    if (coordinate[0]==coordinate[1]){
        for(int i=0; i<local_col_size; i++){
            local_D[i]=local_A[i*local_col_size+i];
            local_R[i*local_col_size+i]=0;
        }
        // send from diag
        if (coordinate[0]!=0){
            MPI_Send(local_D,  local_col_size, MPI_DOUBLE, 0 ,1 ,comm_row);
        }
    }
    // col 0 receive from diag
    if (coordinate[1]==0 && coordinate[0]!=0){
        MPI_Recv(local_D, local_row_size, MPI_DOUBLE, coordinate[0], 1, comm_row, MPI_STATUS_IGNORE);  
    }

    for (int iter=0; iter<max_iter; iter++){
        double l2_norm;
        double* local_y = (double*)malloc(sizeof(double)*local_row_size);

        // update X
        distributed_matrix_vector_mult(n, local_R, local_x, local_y, comm);
        if (coordinate[1] == 0)
        {
            for(int i=0; i<local_row_size; i++){
                local_x[i] = (local_b[i] - local_y[i])/local_D[i];
            }
        }

        // check convergence
        distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);
        if (coordinate[1] == 0)
        {
            double tmp_l2_norm = 0.0;
            for(int i=0; i<local_row_size; i++){
                tmp_l2_norm += pow(local_y[i] - local_b[i],2);
            }
            MPI_Allreduce(&tmp_l2_norm, &l2_norm, 1, MPI_DOUBLE, MPI_SUM, comm_col);
        }
        free(local_y);
        if (coordinate[1]==0){l2_norm = sqrt(l2_norm);}
        MPI_Bcast(&l2_norm, 1, MPI_DOUBLE, 0, comm_row);
        if (l2_norm <= l2_termination){
            break;}
    }

    free(local_D);
    free(local_R);
    MPI_Comm_free(&comm_col);
    MPI_Comm_free(&comm_row);
    return;
}

// wraps the distributed matrix vector multiplication
void mpi_matrix_vector_mult(const int n, double* A,
                            double* x, double* y, MPI_Comm comm)
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_x = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &x[0], &local_x, comm);

    // allocate local result space
    double* local_y = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);

    // gather results back to rank 0
    gather_vector(n, local_y, y, comm);
    delete[] local_y;
}

// wraps the distributed jacobi function
void mpi_jacobi(const int n, double* A, double* b, double* x, MPI_Comm comm,
                int max_iter, double l2_termination)
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_b = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &b[0], &local_b, comm);

    // allocate local result space
    double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);

    // gather results back to rank 0
    gather_vector(n, local_x, x, comm);
    delete[] local_x;
}
