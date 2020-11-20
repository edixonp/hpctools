#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "mkl_lapacke.h"

//Inicializa una matrix con valores aleatorios
void *generate_matrix(int size)
{
    int i;
    double *matrix = (double *)malloc(sizeof(double) * size * size);
    srand(1);
    for (i = 0; i < size * size; i++)
    {
        matrix[i] = rand() % 100;
    }
    return matrix;
}

//Genera una matrix simetrica positiva
void *spd_generate_matrix(int n, int *info) {
    int i;
    double *matrix = generate_matrix(n);
    double *matrix_spd = (double *)malloc(sizeof(double) * n * n);
    for(int i=0; i<n; i++) {
       for(int j=0; j<n; j++) {
          double sum = 0;
	  for(int k=0; k<n; k++) {
	     sum += matrix[i*n+k] * matrix[j*n+k];
	  }
	  matrix_spd[i*n+j] = sum;
             
	}
     }
     free(matrix);
     return matrix_spd;
}

//Imprime los valores de la matrix
void print_matrix(const char *name, double *matrix, int size)
{
    int i, j;
    printf("matrix: %s \n", name);
    for (i = 0; i < size; i++)
    {
       for (j = 0; j < size; j++)
       {
          printf("%f ", matrix[i * size + j]);
       }
       printf("\n");
    }
}

//Se verifica los resultados
int check_result(double *bref, double *b, double eps, int size) {
    int i;
    for ( i=0; i < size * size; i++) {
        if ( abs(bref[i] - b[i]) > eps) 
 	return 0;
    }
    return 1;
}

int  my_dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb) {
	cholesky(a, n);
        soluvect(a, b, n);
        LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, a, lda, ipiv, b, ldb);

}

//Metodo de Cholesky para factorizar una matriz simetrica
void cholesky (double *a, int size) {
    int i, j, k;
    double x, y;
    for (j = 0; j < size; ++j) {
        x = 0.0;
        for (i = 0; i < j; ++i) {
            a[i+j*size] = 0.0;
            x += a[j+i*size] * a[j+i*size];
        }
        x = sqrt(a[j+j*size]-x);
        a[j+j*size] = x;
        for (i = j+1; i < size; ++i) {
            y = 0.0;
            for (k = 0; k < j; ++k)
                y += a[i+k*size] * a[j+k*size];
                a[i+j*size] = (a[i+j*size] - y) / x;
        }
    }
}

void soluvect (const double *a, double *b, int size) {
    int i, j, k;
    for (i = 0; i < size; ++i) {
        for (j = 0; j < size; ++j) {
            b[i+j*size] /= a[i+i*size];
            for (k = i+1; k < size; ++k) 
                b[k+j*size] -= b[i+j*size] * a[k+i*size];
              
         }
    }
    for (j = 0; j < size; ++j) {
        for (i = size-1; i >= 0; --i) {
            for (k = i+1; k < size; ++k) 
                b[i+j*size] -= a[k+i*size] * b[k+j*size];
             
             b[i+j*size] /= a[i+i*size];
        }
    }
}

void m_dmatcop(double *A, int n) {
	for (int i = 0; i < n; i++)
        for (int j = 0; j<i; j++) {
            double c = A[i*n + j];
            A[i*n + j] = A[j*n + i];
            A[j*n + i] = c;
        }
}

void main(int argc, char *argv[])
{
	int size = atoi(argv[1]);
	double *a, *aref;
	double *b, *bref;
	int info;

	a = spd_generate_matrix(size, &info);
	aref = generate_matrix(size);        
	b = generate_matrix(size);
	bref = generate_matrix(size);
	
	memcpy( aref, a, sizeof(double) * size * size);	
	memcpy( bref, b, sizeof(double) * size * size);	
	m_dmatcop(bref, size);

	//print_matrix("A", a, size);
	//print_matrix("B", b, size);

	// Using MKL to solve the system
	MKL_INT n = size, nrhs = size, lda = size, ldb = size;
	MKL_INT *ipiv = (MKL_INT *)malloc(sizeof(MKL_INT)*size);

	clock_t tStart = clock();
	info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, aref, lda, ipiv, bref, ldb);
	printf("Tiempo tomado por MKL: %.6fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

	tStart = clock();    
	MKL_INT *ipiv2 = (MKL_INT *)malloc(sizeof(MKL_INT)*size);        
	my_dgesv(n, nrhs, a, lda, ipiv2, b, ldb);
	printf("Tiempo tomado por mi implementación: %.6fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	
	// transponer la matriz de resultados de la función mkl LAPACKE_dgesv
	m_dmatcop(bref, size);
	
	double eps = 0.000001;
	if (check_result(bref, b, eps, size)==1)
		printf("El resultado esta bien!\n");
	else    
		printf("El resultado es erroneo!\n");
	
	//print_matrix("X", b, size);
	//print_matrix("Xref", bref, size);
	//print_matrix("aref", a, size);

	return 0;
}

