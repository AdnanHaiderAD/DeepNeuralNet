#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "ANN.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "/Users/adnan/NeuralNet/CBLAS/include/cblas.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static ActFunKind *actfunLists;
static int *hidUnitsPerLayer;
static int numHiddenLayers;
static int inputDim;
static OutFuncKind outputfunc;
static ADLINk anndef;

//--------------------------------------------------------------------------------
void initialiseFeaElem(FELink feaElem, FELink srcElem){
	feaElem->xfeaMat = srcElem->yfeatMat;
	feaElem->yfeatMat = NULL;
}

double drand()   /* uniform distribution, (0..1] */
{return (rand()+1.0)/(RAND_MAX+1.0);
}
/* performing the Box Muller transform to map two numbers 
generated from a uniform distribution to a number from a normal distribution centered at 0 with standard deviation 1 */
double random_normal() {
  return sqrt(-2*log(drand())) * cos(2*M_PI*drand());
}

void initialiseBias(double *biasVec,int dim, int srcDim){
	int i;
	for ( i = 0; i<dim;i++){
		randm = random_normal();
		biasVec[i] = randm*(1/sqrt(srcDim));
	}
}
/*the srcDim determines the fan-in to the hidden and output units. The weights ae initialised 
to be inversely proportional to the sqrt(fanin)
*/ 
void initialiseWeights(double *weightMat,int length,int srcDim){
	int i;
	double randm;
	for ( i = 0; i<(length);i++){
		randm = random_normal();
		weightMat[i] = randm*1/(sqrt(srcDim));
	}
	
}
void initialiseLayer(LELink layer,int i, LELink srcLayer){
	int numOfElems;
	
	layer->layerId = i;
	layer->dim = hidUnitsPerLayer[i-1];
	layer->actfuncKind = actfunLists[i-1];
	layer->src = srcLayer;
	layer->srcDim = srcLayer->dim;
	
	/*initialise the seed only once then initialise weights and bias*/
	srand((unsigned int)time(NULL));
	numOfElems = (layer->dim) *(layer->srcDim);
	layer-> weights = malloc(sizeof(double)*numOfElems);
	assert(layer->weights!=NULL);
	initialiseWeights(layer->weightMat,numOfElems,layer->srcDim);
	
	layer->bias = malloc(sizeof(double)*(layer->dim));
	assert(layer->bias!=NULL);
	initialiseBias(bias,layer->dim, layer->srcDim);

	layer->feaElem = (FELink)malloc(sizeof(FeaELem));
	assert(layer->feaElem!=NULL);
	initialiseFeaElem(layer->feaElem,layer->srcLayer->feaElem);
	if (i == (numHiddenLayers-1)){
		layer->type = OUTPUT;
	}else{
		layer->type = HIDDEN;
	}
	layer->errElem = NULL;
}

void initialiseInputLayer(LELink inputlayer){
	layer->id = 0;
	layer->dim = inputDim;
	layer->actfuncKind = IDENTITY;
	layer->src = NULL;
	layer->weights = NULL;
	layer->bias = NULL;
	layer->type = INPUT;
	layer->errElem = NULL;
	
	layer->feaElem = (FELink) malloc(sizeof(FeaELem));
	assert(layer->FeaELem!=NULL);
	layer->feElem->xfeaMat= NULL;//this line may need to be changed later
	layer->feElem->yfeatMat = malloc(sizeof(double)*(layer->dim));
	assert(layer->feaElem->yfeatMat!=NULL);
}

void  initialiseANN(){
	int i;
	anndef = malloc(sizeof(ANNDef));
	assert(anndef!=NULL);

	anndef->outfunc = outputfunc;
	anndef->layerNum = numHiddenLayers;
	for(i = 0; i<anndef->layerNum; i++){
		if (i = 0 ){
			anndef->layerList[i] = (LELink) malloc (sizeof(LayerElem));
			asset(anndef->layerList[i]!=NULL);
			initialiseInputLayer(anndef->layerList[i]);
		}else{
			anndef->layerList[i] = (LELink) malloc (sizeof(LayerElem));
			asset(anndef->layerList[i]!=NULL);
			initialiseLayer(anndef->layerList[i],i, anndef->layerList[i-1]);
		}	
	}

}
//----------------------------------------------------------------------------------------------
double computeTanh(double x){
	return 2*(computeSigmoid(2*x))-1;
}

double computeSigmoid(double x){
	double result;
	result = 1/(1+ exp(-1*x));
	return result;
}

void *computeActOfLayer(double* yfeatMat, int dim,  ActFunKind actfunc){
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
}

void computeLinearActivation(FELink layer){
	layer->feaElem->yfeatMat = malloc(sizeof(double)*(layer->dim));
	assert(layer->feaElem->yfeatMat!=NULL);
	/* y = a W*x + By- Here B=0 and a =1*/
	cblas_dgemv(CblasRowMajor,CblasNoTrans,layer->dim,layer->srcDim,1,layer->weights,layer->srcDim,layer->feaElem->xfeaMat,1,0,layer->feaElem->yfeatMat,1);
	/*y = y+ b- adding the bias*/
	cblas_daxpy(dim,1,layer->bias,1,layer->feaElem->yfeatMat,1);
	
}

void fwdPassOfANN(){
	LELink layer;
	int i;
	switch(anndef->outfunc){
			case REGRESSION:
				for (i = 1; i< anndef->layerNum;i++){
					layer = anndef->layerList[i];
					computeLinearActivation(layer);
					if (i !=(layerNum-1)){
						computeActOfLayer(layer->feaElem->yfeatMat,layer->dim,layer->actfuncKind);
					}
				}
				break;
			case CLASSIFICATION:
				for (i = 1; i< anndef->layerNum;i++){
					layer = anndef->layerList[i];
					computeLinearActivation(layer);
					computeActOfLayer(layer->feaElem->yfeatMat,layer->dim,layer->actfuncKind);
				}
				break;	
	}
}

