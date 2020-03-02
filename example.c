#include "stdlib.h"
#include "stdio.h"
#include <malloc.h>
#include <omp.h>
#include <mpi.h>
#include <math.h>
#include <emmintrin.h>

#define SM (CLS / sizeof (double))

#define MAXT 3 // set this variable to the number of times to be measured per rank

void print_results(double *a, double *b, double *r1, double *r2, int N, int M);

void main(int argc, char **argv){

	MPI_Init(&argc, &argv);
	int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	
	int i, j, k, i2, j2, k2;

	int N = atoi(argv[1]);
	int M = (argc>2)? atoi(argv[2]) : N;

	/** MEMMORY ALLOCATION **/

	double *a = (double*)_mm_malloc(sizeof(double)*N*N,64);
	double *b = (double*)_mm_malloc(sizeof(double)*N*M,64);	
	double *c1 = (double*)_mm_malloc(sizeof(double)*N*M,64);
	double *c2 = (double*)_mm_malloc(sizeof(double)*N*M,64);

	double times[MAXT];
	double *t = &times[0];

	/** INITIALIZATION **/

	*t = omp_get_wtime();
	#pragma omp parallel for 
	for (i=0; i<N*N; i++)
	{
		a[i] = sin((double)(i%20));
	}
	#pragma omp parallel for 
	for (i=0; i<N*M; i++)
	{
		b[i] = cos((double)(i%20));
		c1[i] = 0.;
		c2[i] = 0.;
	}
	*t = omp_get_wtime() - *t;
	t++;

	/** FIRST MATRIX-MATRIX OPERATION: A SIMPLE MULTIPLICATION**/
	// memmory alignment is assumed

	*t = omp_get_wtime();
  #pragma omp parallel for private(j,k)
  for (i=0; i<N; i++)
  {
    for (j=0; j<M; j++)
		#pragma omp simd
      for (k=0; k<N; k++)
        c1[i*M+j] += a[i*N+k]*b[j*M+k];
  }
	*t = omp_get_wtime() - *t;
  t++;

	/** SECOND MATRIX-MATRIX OPERATION: A VECTORIZED MULTIPLICATION **/
	// https://people.freebsd.org/~lstewart/articles/cpumemory.pdf (see section 6.2.1, Appendix A.1)

	double *rres;
	double *rmul1;
	double *rmul2;

	*t = omp_get_wtime();
	#pragma omp parallel for private(j,k,i2,j2,k2)
	for (i = 0; i < N; i += SM)
		for (j = 0; j < M; j += SM)
			for (k = 0; k < N; k += SM)
				for (i2 = 0, rres = &c2[i*M+j], rmul1 = &a[i*N+k]; i2 < SM;
						 ++i2, rres += M, rmul1 += N)
				{
					_mm_prefetch (&a[8], _MM_HINT_NTA);
					for (k2 = 0, rmul2 = &b[k*M+j]; k2 < SM; ++k2, rmul2 += M)
					{
						__m128d m1d = _mm_load_sd (&rmul1[k2]);
						m1d = _mm_unpacklo_pd (m1d, m1d);
						for (j2 = 0; j2 < SM; j2 += 2)
						{				
							__m128d m2 = _mm_load_pd (&rmul2[j2]);
							__m128d r2 = _mm_load_pd (&rres[j2]);
							_mm_store_pd (&rres[j2],
							_mm_add_pd (_mm_mul_pd (m2, m1d), r2));
						}
					}
	}
	*t = omp_get_wtime() - *t;

	/** COMMUNICATE COMPUTATIONAL TIMES **/
	double timesglobal[MAXT];

	MPI_Reduce(&times, &timesglobal, MAXT, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	if(rank==0)
	{
		for(i=0;i<MAXT; i++) printf("T%d %1.3f ", i, timesglobal[i]);
		printf("\n");
	}
	
	/** PRINT RESULTS IF REQUIRED IN COMPILATION TIME **/

#ifdef PRINT_RESULTS
	print_results(a,b,c1,c2,N,M);
#endif

	/** END THE PROGRAM **/

	_mm_free(a); _mm_free(b);
	_mm_free(c1); _mm_free(c2);

	MPI_Finalize();
}

void print_results(double *a, double *b, double *r1, double *r2, int N, int M)
{
	int i,j;
	//a	
	printf("\n\n");
	for(i=0; i<N; i++)
	{
		for(j=0; j<N; j++)
			printf("%1.2f\t", a[i*N+j]);
		printf("\n");
	}
	printf("\n\n");

	//b
	printf("\n\n");
	for(i=0; i<N; i++)
	{
		for(j=0; j<M; j++)
			printf("%1.2f\t", b[i*M+j]);
		printf("\n");
	}
	printf("\n\n");
	
	//c1
	printf("\n\n");
	for(i=0; i<N; i++)
	{
		for(j=0; j<M; j++)
			printf("%1.2f\t", r1[i*M+j]);
		printf("\n");
	}
	printf("\n\n");

	//c2
	printf("\n\n");
	for(i=0; i<N; i++)
	{
		for(j=0; j<M; j++)
			printf("%1.2f\t", r2[i*M+j]);
		printf("\n");
	}
	printf("\n\n");
	
}
