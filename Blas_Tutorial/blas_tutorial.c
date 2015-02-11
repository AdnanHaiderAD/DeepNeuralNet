#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "/Users/adnan/NeuralNet/CBLAS/include/cblas.h"

typedef enum {SIGMOID,IDENTITY,TANH} ActFunKind;

double computeTanh(double x);
double computeSigmoid(double x);
double *computeActOfLayer(double* yfeatMat, int dim,  ActFunKind actfunc);
void testfunct(double *v);
double *initialiseWeights(int rows, int columns,int srcDim);
double *initialiseBias(int dim, int srcDim);


static double *arr;

int main(){

	int i;
	int dim;
	int srcDim;
	int size;
	double weights[] ={ 2, 0,0,2,1,0,0,2};

	dim = 4;
	srcDim = 2;
	double *linearActivation = malloc(sizeof(double)*dim);
	
	double vector[] = { 1,1,1};
	double test3[] ={1,1};
	double bias[] = { 0,0,1,1};

	double vec[] ={2,2,2};
	double test_vec[] ={ 3 ,3,3};

	
	/*matrix*/
	double A[] ={ 3, 0, 0,0,3,0,0,0,3};
	
	

	float dotproduct;
	double result;
	double norm;

	printf("reaches here 1\n");
	arr = (double *) malloc(sizeof(double)*3);
	testfunct(vector);
	for(i = 0; i<srcDim;i++){
		printf(" the output is %d  %lf\n",i,arr[i] );
	}
	printf("reaches here 2 \n");

	free(arr);

	printf("reaches here 3\n");

	cblas_dgemv(CblasRowMajor,CblasNoTrans,dim,srcDim,1,weights,srcDim,test3,1,0,linearActivation,1);
	cblas_daxpy(dim,1,bias,1,linearActivation,1);
	size = sizeof(linearActivation)/sizeof(*linearActivation);
	printf("The size of the activation function is %d \n",size );
	for(i = 0; i<dim;i++){
		printf(" the output is %d  %lf\n",i,linearActivation[i] );
	}
	linearActivation = computeActOfLayer(linearActivation,dim,SIGMOID);
	for(i = 0; i<dim;i++){
		printf(" the output is %d  %lf\n",i,linearActivation[i] );
	}

	linearActivation = computeActOfLayer(linearActivation,dim,TANH);
	for(i = 0; i<dim;i++){
		printf(" the output is %d  %lf\n",i,linearActivation[i] );
	}


	free(linearActivation);


	
	dotproduct = cblas_ddot(3,vec,0,vec,0);
	result = cblas_ddot(3,vec,0,vec,0);
	norm = cblas_dnrm2(3,vec,1);
	//cblas_daxpy(1,1,vec,0,test_vec,0);


	/* y = a A*x + by*/
	cblas_dgemv(CblasRowMajor,CblasNoTrans,3,3,1,A,3,vector,1,1,vector,1);
	//cblas_dscal(3, 4.323, vector, 1);

	for (i =0; i < 3; i++){
		printf( "x[%d] = %lf ",i,vector[i]);

	}
	printf("The dot product is %f\n", dotproduct);
	printf("The second dot product value is %lf\n", result);
	printf("The norm of the first vector is %lf\n",norm);
	printf("the value should 4  so %lf\n",vector[0]);

 }

 double computeTanh(double x){
	return 2*(computeSigmoid(2*x))-1;
}

double computeSigmoid(double x){
	double result;
	result = 1/(1+ exp(-1*x));
	return result;
}

double *computeActOfLayer(double* yfeatMat, int dim,  ActFunKind actfunc){
	int i = 0;
	switch(actfunc){
		case SIGMOID:
			for (i = 0;i < dim;i++){
				yfeatMat[i] = computeSigmoid(yfeatMat[i]);
			}
			break;
		case TANH:
			for(i = 0; i< dim; i++){
				yfeatMat[i] = computeTanh(yfeatMat[i]);
			}
			break;	
		default:
			break;	
	}
	return yfeatMat;
}

double *initialiseBias(int dim, int srcDim){
	int i;
	double randm;
	double* biasVec;
	weighMat =  malloc(sizeof(double)*(dim));
	srand((unsigned int)time(NULL));
	for ( i = 0; i<dim;i++){
		randm = double(rand())/double(RAND_MAX);
		biasVec[i] = randm*sqrt(srcDim)
	}
	return biasVec;

}
/*the srcDim determines the fan-in to the hidden and output units. The weights ae initialised 
to be inversely proportional to the sqrt(fanin)
*/ 
double *initialiseWeights(int rows, int columns,int srcDim){
	int i;
	double randm;
	double* weighMat;
	weighMat =  malloc(sizeof(double)*(rows*columns));
	srand((unsigned int)time(NULL));
	for ( i = 0; i<(rows*columns);i++){
		randm = double(rand())/double(RAND_MAX);
		weighMat[i] = randm*sqrt(srcDim)
	}
	return weighMat;
}





void testfunct(double *v){
	int i;
	double *value;
	value = computeActOfLayer(v,3,SIGMOID);
	for (i =0; i<3 ;i++){
		arr[i] = value[i];
	}


}