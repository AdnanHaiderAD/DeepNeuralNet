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
static int numLayers;
static int inputDim;
static OutFuncKind outputfunc;
static ADLink anndef;

//--------------------------------------------------------------------------------
/**this section of the src code deals with initialisation of ANN **/

void initialiseFeaElem(FELink feaElem, FELink srcElem){
	feaElem->xfeatMat = srcElem->yfeatMat;
	
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
	double randm;
	for ( i = 0; i<dim;i++){
		randm = random_normal();
		biasVec[i] = randm*(1/sqrt(srcDim));
	}
}
/*the srcDim determines the fan-in to the hidden and output units. The weights ae initialised 
to be inversely proportional to the sqrt(fanin)
*/ 
void initialiseWeights(double *weights,int length,int srcDim){
	int i;
	double randm;
	for ( i = 0; i<(length);i++){
		randm = random_normal();
		weights[i] = randm*1/(sqrt(srcDim));
	}
	
}
void initialiseLayer(LELink layer,int i, LELink srcLayer){
	int numOfElems;
	
	layer->id = i;
	layer->dim = hidUnitsPerLayer[i-1];
	layer->actfuncKind = actfunLists[i-1];
	layer->src = srcLayer;
	layer->srcDim = srcLayer->dim;
	
	/*initialise the seed only once then initialise weights and bias*/
	srand((unsigned int)time(NULL));
	numOfElems = (layer->dim) *(layer->srcDim);
	layer-> weights = malloc(sizeof(double)*numOfElems);
	assert(layer->weights!=NULL);
	initialiseWeights(layer->weights,numOfElems,layer->srcDim);
	
	layer->bias = malloc(sizeof(double)*(layer->dim));
	assert(layer->bias!=NULL);
	initialiseBias(layer->bias,layer->dim, layer->srcDim);

	layer->feaElem = (FELink)malloc(sizeof(FeaElem));
	assert(layer->feaElem!=NULL);
	initialiseFeaElem(layer->feaElem,layer->src->feaElem);
	layer->feaElem->yfeatMat = malloc(sizeof(double)*(layer->dim));

	if (i == (numLayers-1)){
		layer->type = OUTPUT;
	}else{
		layer->type = HIDDEN;
	}
	layer->errElem = NULL;
}

void initialiseInputLayer(LELink layer){
	layer->id = 0;
	layer->dim = inputDim;
	layer->actfuncKind = IDENTITY;
	layer->src = NULL;
	layer->weights = NULL;
	layer->bias = NULL;
	layer->type = INPUT;
	layer->errElem = NULL;
	
	layer->feaElem = (FELink) malloc(sizeof(FeaElem));
	assert(layer->feaElem!=NULL);
	layer->feaElem->xfeatMat= NULL;
	layer->feaElem->yfeatMat = malloc (sizeof(double)*inputDim);//this line may need to be changed later
	assert(layer->feaElem->yfeatMat!=NULL);
}	

void  initialiseANN(){
	int i;
	anndef = malloc(sizeof(ANNDef));
	assert(anndef!=NULL);
	
	anndef->outfunc = outputfunc;
	anndef->layerNum = numLayers;
	anndef->layerList = malloc (sizeof(LELink)*numLayers);
	assert(anndef->layerList!=NULL);
	for(i = 0; i<anndef->layerNum; i++){
		if (i == 0 ){
			anndef->layerList[i] = malloc (sizeof(LayerElem));
			assert (anndef->layerList[i]!=NULL);
			initialiseInputLayer(anndef->layerList[i]);
			
		}else{
			anndef->layerList[i] = (LELink) malloc (sizeof(LayerElem));
			assert(anndef->layerList[i]!=NULL);
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

void computeActOfLayer(double* yfeatMat, int dim,  ActFunKind actfunc){
	int i ;
	printf("THE DIM IS %d\n",dim);
	switch(actfunc){
		case SIGMOID:
			for (i = 0;i < dim;i++){
				printf("Value of linear activation %f \n", yfeatMat[i]);
				yfeatMat[i] = computeSigmoid(yfeatMat[i]);
				printf("value of activation %f \n",yfeatMat[i]);
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

void computeLinearActivation(LELink layer){
	if (layer->dim > 1){
		/* y = a W*x + By- Here B=0 and a =1*/
		cblas_dgemv(CblasRowMajor,CblasNoTrans,layer->dim,layer->srcDim,1,layer->weights,layer->srcDim,layer->feaElem->xfeatMat,1,0,layer->feaElem->yfeatMat,1);
		/*y = y+ b- adding the bias*/
		cblas_daxpy(layer->dim,1,layer->bias,1,layer->feaElem->yfeatMat,1);
	}else{
		layer->feaElem->yfeatMat[0] = cblas_ddot(layer->src->dim,layer->weights,1,layer->feaElem->xfeatMat,1);
		layer->feaElem->yfeatMat[0] = layer->feaElem->yfeatMat[0]+layer->bias[0];
	}	
}

void fwdPassOfANN(){
	LELink layer;
	int i;
	switch(anndef->outfunc){
			case REGRESSION:
				for (i = 1; i< anndef->layerNum;i++){
					layer = anndef->layerList[i];
					computeLinearActivation(layer);
					computeActOfLayer(layer->feaElem->yfeatMat,layer->dim,layer->actfuncKind);
					
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

void freeMemory(){
	int i;
	if (anndef != NULL){
		for (i = 0;i<numLayers;i++){
			if (anndef->layerList[i] !=NULL){
				if (anndef->layerList[i]->feaElem != NULL){
					if (anndef->layerList[i]->feaElem->yfeatMat !=NULL){
						free (anndef->layerList[i]->feaElem->yfeatMat);
					}
					free(anndef->layerList[i]->feaElem);
				}
				if (anndef->layerList[i]->weights !=NULL){
					free (anndef->layerList[i]->weights);
				}
				if (anndef->layerList[i]->bias !=NULL){
					free (anndef->layerList[i]->bias);
				}
				free (anndef->layerList[i]);
			}
		}
		free(anndef->layerList);	
		free(anndef);
		
	}	
}

//=================================================================================

int main(){
	int i;
	/*testing forward pass of ANN*
	
	Test 1 : with single input 

	The structure of ANN is 3 layers : input layer dim =2, hidden layer dim =3, 
	output layer dim =1, the output of ANN is regression
	*/
	//initialise
	ActFunKind list[] = {SIGMOID,SIGMOID};
	actfunLists = list;
	int arr[] ={3,1};
	hidUnitsPerLayer = arr;
	numLayers = 3; ;
	inputDim = 2;
	outputfunc = CLASSIFICATION;
	printf("Debug : 1 \n");
	initialiseANN();

	anndef->layerList[0]->feaElem->yfeatMat[0] = 1;
	anndef->layerList[0]->feaElem->yfeatMat[1] = 1;

	/** printing the output of the hidden and output units **/
	fwdPassOfANN();

	/** printin out the weights and bias of the hidden layer**/
	printf("1st check %d\n",(anndef->layerList[0]->dim*anndef->layerList[1]->dim));
	printf("2nd check %d\n",(anndef->layerList[0]->dim -1));
	for (i = 0; i < (anndef->layerList[0]->dim*anndef->layerList[1]->dim);i+=(anndef->layerList[0]->dim )){
		printf("the weights to  hidden unit is %f  %f \n ",anndef->layerList[1]->weights[i],anndef->layerList[1]->weights[i+1]);

	}
	printf("the bias to the first hidden layer is %f %f %f \n",anndef->layerList[1]->bias[0],anndef->layerList[1]->bias[1],anndef->layerList[1]->bias[2]);
	
	/**printing out the outputs of the hidden layer*/
	for (i =0 ; i <anndef->layerList[1]->dim;i+=anndef->layerList[1]->dim){
		printf("the outputs for hidden units is %f ,%f, %f \n\n",anndef->layerList[1]->feaElem->yfeatMat[i],anndef->layerList[1]->feaElem->yfeatMat[i+1],anndef->layerList[1]->feaElem->yfeatMat[i+2]);
	}	

	/**printing out the weights and bias associated with the output layer **/
	for (i = 0; i < anndef->layerList[2]->dim*anndef->layerList[1]->dim;i+=anndef->layerList[1]->dim){
	printf ( "the weights to the output is %f,%f,%f \n",anndef->layerList[2]->weights[i],anndef->layerList[2]->weights[i+1],anndef->layerList[2]->weights[i+2]);
	}
	printf ("the bias to the final output is  %f \n ",anndef->layerList[2]->bias[0]);
	
	
	for (i =0 ;i<anndef->layerList[2]->dim;i++){
	printf("The output of ANN is %f\n",anndef->layerList[2]->feaElem->yfeatMat[i]);
	}




	freeMemory();

	




}

