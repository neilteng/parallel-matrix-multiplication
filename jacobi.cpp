/**
 * @file    jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements matrix vector multiplication and Jacobi's method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
#include "jacobi.h"

/*
 * TODO: Implement your solutions here
 */

// my implementation:
#include <iostream>
#include <math.h>

// Calculates y = A*x for a square n-by-n matrix A, and n-dimensional vectors x
// and y
void matrix_vector_mult(const int n, const double* A, const double* x, double* y)
{
    double tmp;
    for(int i=0;i<n;i++){
        tmp = 0;
        for(int j=0;j<n;j++){
            tmp += A[n*i+j]*x[j];
        }
        y[i]=tmp;
    }
}

// Calculates y = A*x for a n-by-m matrix A, a m-dimensional vector x
// and a n-dimensional vector y
void matrix_vector_mult(const int n, const int m, const double* A, const double* x, double* y)
{
    double tmp;
    for(int i=0;i<n;i++){
        tmp = 0;
        for(int j=0;j<m;j++)
            tmp += A[m*i+j]*x[j];
        y[i]=tmp;
    }
}

// implements the sequential jacobi method
void jacobi(const int n, double* A, double* b, double* x, int max_iter, double l2_termination)
{
    int iter=0;
    double D[n];
    double R[n*n];
    //Compute R, D and initial x.
    for (int i = 0; i < n; i++) {
        x[i] = 0;
        for (int j = 0; j < n; j++) {
            if (i == j) { 
                D[i] = A[i * n + i];
                R[i * n + j] = 0; 
            }
            else 
                R[i * n + j] = A[i * n + j]; 
        }
    }
    while (iter < max_iter) {
        double x_tmp[n];
        double l2=0;
        iter++;
        //compute the new x
        matrix_vector_mult(n, R, x, x_tmp);
        for (int i = 0; i < n; i++)
            x[i] = (b[i] - x_tmp[i]) / D[i];
        // detect termination,
        matrix_vector_mult(n, A, x, x_tmp);
        for (int i = 0; i < n; i++)
            l2 += (x_tmp[i] - b[i]) * (x_tmp[i] - b[i]);
        if (sqrt(l2) < l2_termination) 
            break;
    }
}
