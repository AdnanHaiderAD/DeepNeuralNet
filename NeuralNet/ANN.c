#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "ANN.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "/Users/adnan/NeuralNet/CBLAS/include/cblas.h"

static ActFunKind *actfunLists;
static int *hidUnitsPerLayer;
static int numHiddenLayers;
static int inputDim;
static OutFuncKind outputfunc;
static ADLINk anndef;

//--------------------------------------------------------------------------------
FELink initialiseFeaElem(int feaDim, FELink srcFeaElem){
	FELINK feaElem;
	feaElem = malloc(sizeof(FeaELem));
	asset(feaElem !=NULL);

	feaElem->xfeaMat = srcElem->yfeatMat;
	feaElem->yfeatMat = NULL;
	return feaElem;

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

LELink initialiseLayer(int i, LELink srcLayer){
	LELink layer;
	layer = malloc (sizeof(LayerElem));
	assert(layer != NULL);

	layer->layerId = i;
	/*the input layer setup is defined separately so the first hidden layer is indexed at 0*/
	layer->dim = hidUnitsPerLayer[i-1];
	layer->actfuncKind = actfunLists[i-1];

	layer->src = srcLayer;
	layer->srcDim = srcLayer->dim;
	layer-> weights = initialiseWeights(layer->dim,layer->srcDim);
	layer-> bias = initialiseBias(layer->dim);
	layer->feaElem = initialiseFeaElem(layer->dim,layer->srcLayer->feaElem);
	if (i == (numHiddenLayers-1)){
		layer->type = OUTPUT;
	}else{
		layer->type = HIDDEN;
	}
	layer->errElem = NULL;
	
	return layer;
}

void initialiseInputLayer(){
	FELink inputLayer;
	FELink feElem;
	inputLayer = malloc(sizeof(LayerElem));
	assert(layer!=NULL);

	layer->id = 0;
	layer->dim = inputDim;
	layer->actfuncKind = IDENTITY;
	layer->src = NULL;
	layer->weights = NULL;
	layer->bias = NULL;
	layer->type = INPUT;
	
	feElem = malloc(sizeof(FeaELem));
	feElem->xfeaMat= NULL;
	feElem->yfeatMat = malloc(sizeof(double)*(layer->dim));
	layer->feaElem = feElem;
	layer->errElem = NULL;
	return layer;
}

void  initialiseANN(){
	int i;
	anndef = malloc(sizeof(ANNDef));
	assert(anndef!=NULL);

	anndef->outfunc = outputfunc;
	anndef->layerNum = numHiddenLayers;
	for(i = 0; i<anndef->layerNum; i++){
		if (i = 0 ){
			anndef->layerList[i] = initialiseInputLayer();
		}else{
			anndef->layerList[i] = initialiseLayer(i, anndef->layerList[i-1]);
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

double *computeLinearActivation(FELink layer){
	double *linearActivation;
	linearActivation = malloc[sizeof(double)*(layer->dim)];
	/* y = a W*x + By- Here B=0 and a =1*/
	cblas_dgemv(CblasRowMajor,CblasNoTrans,layer->dim,layer->srcDim,1,layer->weights,layer->srcDim,layer->feaElem->xfeaMat,1,0,linearActivation,1);
	/*y = y+ b- adding the bias*/
	cblas_daxpy(dim,1,layer->bias,1,linearActivation,1);
	return linearActivation;
}

void fwdPassOfANN(){
	LELink layer;
	int i;
	double *yfeatMat;

	switch(anndef->outfunc){
			case REGRESSION:
				for (i = 1; i< anndef->layerNum;i++){
					layer = anndef->layerList[i];
					yfeatMat = computeLinearActivation(layer);
					if (i !=(layerNum-1)){
						layer->feaElem->yfeatMat = computeActOfLayer(yfeatMat,layer->dim,layer->actfuncKind);
					}else{
						layer->feaElem->yfeatMat = yfeatMat;
					}
				}
				break;
			case CLASSIFICATION:
				for (i = 1; i< anndef->layerNum;i++){
					layer = anndef->layerList[i];
					yfeatMat = computeLinearActivation(layer);
					layer->feaElem->yfeatMat = computeActOfLayer(yfeatMat,layer->dim,layer->actfuncKind);
				}
				break;	
	}
}

