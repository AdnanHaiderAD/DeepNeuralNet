#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "ANN.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef BATCHSAMPLES
#define BATCHSAMPLES 1
#endif
#ifdef CBLAS
#include "/Users/adnan/NeuralNet/CBLAS/include/cblas.h"
#endif


/*configurations for adapting parameters*/
static double learningrate;
static double weightdecay;
static double momentum;
static int  maxEpochNum;
static double minLR;


static ActFunKind *actfunLists;
static int *hidUnitsPerLayer;
static int numLayers;
static int inputDim;
static int targetDim;
static double *labels;
static OutFuncKind target;
static ObjFuncKind errfunc;
static ADLink anndef;


//--------------------------------------------------------------------------------
/**this section of the src code deals with initialisation of ANN **/
void initialiseErrElems(ADLink anndef){
	int i;
	LELink layer,srcLayer;
	for (i = 0; i < anndef->layerNum ;i++){
		layer = anndef->layerList[i];
		layer->errElem = (ERLink) malloc (sizeof(ErrElem));
		layer->errElem->dxFeatMat = (double *) malloc(sizeof(double)* (layer->dim*layer->srcDim));	
		if ( i!=0){
			srcLayer = layer->src;
			srcLayer->errElem->dyFeatMat = layer->errElem->dxFeatMat;
		}	
	}
	
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
void initialiseWeights(double *weights,int dim,int srcDim){
	int i,j;
	double randm;
	//this is not an efficient way of doing but it allows better readibility
	for (i = 0; i < dim; i++){
		for(j = 0; j < srcDim;j++){
			randm = random_normal();
			printf("value %f\n",randm * 1/(sqrt(srcDim)));
			*weights = randm * 1/(sqrt(srcDim));
			weights = weights + 1;
		}
	}
}
void initialiseLayer(LELink layer,int i, LELink srcLayer){
	int srcDim,numOfElems;
	if (srcLayer != NULL ){
		srcDim = srcLayer->dim;
	}else{
		srcDim = inputDim;
	}

	layer->id = i;
	layer->src = srcLayer;
	layer->srcDim = srcDim;
	
	//setting the layer's role
	if (i == (anndef->layerNum-1)){
		layer->role = OUTPUT;
		layer->dim = targetDim;
	}else{
		layer->role = HIDDEN;
		layer->dim = hidUnitsPerLayer[i];
	}
	layer->actfuncKind = actfunLists[i];

	//for binary classification
	if (layer->role==OUTPUT && layer->dim == 2 && anndef->target==CLASSIFICATION){
		layer->dim = 1; 
	}
	//initialise weights and biases: W is node by feadim Matrix 
	numOfElems = (layer->dim) * (layer->srcDim);
	layer-> weights = malloc(sizeof(double)*numOfElems);
	assert(layer->weights!=NULL);
	initialiseWeights(layer->weights,layer->dim,layer->srcDim);
	
	layer->bias = malloc(sizeof(double)*(layer->dim));
	assert(layer->bias!=NULL);
	initialiseBias(layer->bias,layer->dim, layer->srcDim);
	
	//initialise feaElems
	layer->feaElem = (FELink) malloc(sizeof(FeaElem));
	assert(layer->feaElem!=NULL);
	layer->feaElem->yfeatMat = malloc(sizeof(double)*(layer->dim*BATCHSAMPLES));
	layer->feaElem->xfeatMat = (srcLayer != NULL) ? srcLayer->feaElem->yfeatMat : NULL;

	//intialise traininfo
	layer->info = (TRLink) malloc(sizeof(TrainInfo));
	layer->info->dwFeatMat = malloc(sizeof(double)*numOfElems);
	layer->info->dbFeaMat = malloc(sizeof(double)*layer->dim);

	if (momentum > 0) {
		layer->updateWeightMat = malloc(sizeof(double)*numOfElems);
		layer->updateBiasMat = malloc(sizeof(double)*(layer->dim));
	}
		
}

void  initialiseANN(){
	int i;
	anndef = malloc(sizeof(ANNDef));
	assert(anndef!=NULL);
	
	anndef->target = target;
	anndef->layerNum = numLayers;
	anndef->labelMat = labels;
	anndef->layerList = (LELink *) malloc (sizeof(LELink)*numLayers);
	assert(anndef->layerList!=NULL);
	/*initialise the seed only once then initialise weights and bias*/
	srand((unsigned int)time(NULL));
	//initilaise layers
	for(i = 0; i<anndef->layerNum; i++){
		anndef->layerList[i] = (LELink) malloc (sizeof(LayerElem));
		assert (anndef->layerList[i]!=NULL);
		printf("layer %d \n",i);
		if (i == 0 ){
			initialiseLayer(anndef->layerList[i],i, NULL);
		}else{
			initialiseLayer(anndef->layerList[i],i, anndef->layerList[i-1]);
			if(anndef->layerList[i]->role == OUTPUT) anndef->layerList[i]->info->labelMat = anndef->labelMat;
		}	
	}

	anndef->errorfunc = errfunc;
	//initialise ErrElems of layers for back-propagation
	initialiseErrElems(anndef);
}
//----------------------------------------------------------------------------------------------
/*this section of the code deals with forward propgation of a deep neural net **/

double computeTanh(double x){
	return 2*(computeSigmoid(2*x))-1;
}
double computeSigmoid(double x){
	double result;
	result = 1/(1+ exp(-1*x));
	return result;
}
/*computing non-linear activation*/
void computeActOfLayer(LELink layer){
	int i ;
	double sum;
	switch(layer->role){
		case HIDDEN:
			switch(layer->actfuncKind){
				case SIGMOID:
				for (i = 0;i < layer->dim*BATCHSAMPLES;i++){
					layer->feaElem->yfeatMat[i] = computeSigmoid(layer->feaElem->yfeatMat[i]);
				}
				break;
			case TANH:
				for(i = 0; i< layer->dim*BATCHSAMPLES; i++){
					layer->feaElem->yfeatMat[i] = computeTanh(layer->feaElem->yfeatMat[i]);
				}
				break;	
			default:
				break;	
			}
			break;
		case OUTPUT:
			switch(layer->actfuncKind){
				case SIGMOID:
					if (layer->dim==1){
					/*logistic regression now yfeatmmat is now an array where of one output activation per sample*/
						for (i = 0;i < layer->dim*BATCHSAMPLES;i++){
							layer->feaElem->yfeatMat[i] = computeSigmoid(layer->feaElem->yfeatMat[i]);
						}
					}else{
						printf("ERROR to perform binary classification,the number of non-zero output nodes must be <=2");
						exit(0);
					}
					break;	
				case SOFTMAX:
				//softmax activation
					for (i = 0;i < layer->dim*BATCHSAMPLES;i++){
						layer->feaElem->yfeatMat[i] = exp(layer->feaElem->yfeatMat[i]);
						sum+=layer->feaElem->yfeatMat[i];
					}
					for (i = 0;i < layer->dim*BATCHSAMPLES;i++){
						layer->feaElem->yfeatMat[i] =layer->feaElem->yfeatMat[i]/sum;
					}	
					break;
				default:
					break;	
			}
		default:
			break;
	}
	
}
/* Yfeat is batchSamples by nodeNum matrix(stored as row major)  = X^T(row-major)-batch samples by feaMat * W^T(column major) -feaMat By nodeNum */
void computeLinearActivation(LELink layer){
	#ifdef CBLAS
		int i,off;
		for (i = 0, off = 0; i < BATCHSAMPLES;i++, off += layer->dim){
			 cblas_dcopy(layer->dim, layer->bias, 1, layer->feaElem->yfeatMat + off, 1);
		}
		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, layer->dim, BATCHSAMPLES, layer->srcDim, 1, layer->weights, layer->srcDim, layer->feaElem->xfeatMat, layer->srcDim, 1, layer->feaElem->yfeatMat, layer->dim);
	#endif
}   

/*forward pass*/
void fwdPassOfANN(){
	LELink layer;
	int i;
	switch(anndef->target){
			case REGRESSION:
				for (i = 0; i< anndef->layerNum-1;i++){
					layer = anndef->layerList[i];
					computeLinearActivation(layer);
					computeActOfLayer(layer);
				}
				computeLinearActivation(anndef->layerList[anndef->layerNum-1]);
				break;
			case CLASSIFICATION:
				for (i = 0; i< anndef->layerNum;i++){
					layer = anndef->layerList[i];
					computeLinearActivation(layer);
					computeActOfLayer(layer);
				}
				break;	
	}
}
//----------------------------------------------------------------------------------=
/*This section of the code implements the back-propation algorithm  to compute the error derivatives*/
void computeDrvAct(double *dyfeat , double *yfeat,int len){
	//CPU Version
	int i;
	printf("\n dE/da for layer  \n");
	for (i = 0; i< len;i++){
		dyfeat[i] = dyfeat[i]*yfeat[i];
		printf("%lf ",dyfeat[i] );
	}
}

void computeActivationDrv (LELink layer){
	int i;
	switch (layer->actfuncKind){
		case SIGMOID:
			//CPU verion
		   printf("dz/da for layer %d \n",layer->id);
			for (i = 0; i<layer->dim*BATCHSAMPLES;i++){
				layer->feaElem->yfeatMat[i] = layer->feaElem->yfeatMat[i]*(1-layer->feaElem->yfeatMat[i]);
				printf("%lf ",layer->feaElem->yfeatMat[i] );
			}
		default:
			break;	
	}
}

void sumColsOfMatrix(double *dyFeatMat,double *dbFeatMat,int dim,int batchsamples){
	#ifdef CBLAS
		int i;
		double* ones = malloc (sizeof(double)*batchsamples);
		for (i = 0; i<batchsamples;i++){
			ones[i] = 1;
		}
		//multiply node by batchsamples with batchsamples by 1
		cblas_dgemv(CblasColMajor,CblasNoTrans, dim,batchsamples,1,dyFeatMat,dim,ones,1,0,dbFeatMat,1);
	#endif
}

void subtractMatrix(double *dyfeat, double* labels, int dim){
	//blas routine
	#ifdef CBLAS
		cblas_daxpy(dim,-1,labels,1,dyfeat,1);
		int i;
		for (i = 0; i<dim;i++){
			printf("%lf ", dyfeat[i]);
		}
		printf("\n");

	#else
	//CPU version
		int i;
		//printf("dE/da at the output node \n");
		for (i = 0; i<dim;i++){
			dyfeat[i] = dyfeat[i]-labels[i];
			//printf("%lf ", dyfeat[i]);
		}
	//	printf("\n");
	#endif
}

void CalcOutLayerBackwardSignal(LELink layer,ObjFuncKind errorfunc ){
	switch(errorfunc){
		case (XENT):
			switch(layer->actfuncKind){
				case SIGMOID:
					subtractMatrix(layer->errElem->dyFeatMat,layer->info->labelMat,layer->dim*BATCHSAMPLES);
				break;

				case SOFTMAX:
					subtractMatrix(layer->errElem->dyFeatMat,layer->info->labelMat,layer->dim*BATCHSAMPLES);
				break;

				case TANH:
				break;

				case IDENTITY:
				break;
			}
		default:
		break	;
	}
}

void BackPropBatch(ADLink anndef){
	int i,c,j,k;
	LELink layer;
	for (i = (anndef->layerNum-1); i>=0;i--){
		printf("\nlayer number %d\n",i);
		layer = anndef->layerList[i];
		if (layer->role ==OUTPUT){
			layer->errElem->dyFeatMat = layer->feaElem->yfeatMat;
			CalcOutLayerBackwardSignal(layer,anndef->errorfunc);
		}else{
			// from previous iteration dxfeat that is dyfeat now is dE/dZ.. computing dE/da
			computeActivationDrv(layer); 
			computeDrvAct(layer->errElem->dyFeatMat,layer->feaElem->yfeatMat,layer->dim*BATCHSAMPLES); 

		}
		#ifdef CBLAS
		//compute dxfeatMat: the result  should be an array [ b1 b2..] where b1 is one of dim srcDim
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, layer->srcDim, BATCHSAMPLES, layer->dim, 1, layer->weights, layer->srcDim, layer->errElem->dyFeatMat, layer->dim, 0,layer->errElem->dxFeatMat,layer->srcDim);
		//compute derivative with respect to weights: the result  should be an array of array of [ n1 n2] where n1 is of length srcDim
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, layer->srcDim, layer->dim, BATCHSAMPLES, 1, layer->feaElem->xfeatMat, layer->srcDim, layer->errElem->dyFeatMat, layer->dim, 0, layer->info->dwFeatMat, layer->srcDim);
		//compute derivative with respect to bias: the result should an array of size layer->dim ..we just sum the columns of dyFeatMat
		sumColsOfMatrix(layer->errElem->dyFeatMat,layer->info->dbFeaMat,layer->dim,BATCHSAMPLES);
		#endif

		c = 0;

		printf("\n xfeat for layer %d \n",layer->id);
		for (j = 0; j< layer->srcDim*BATCHSAMPLES;j++){
			printf("%lf ",layer->feaElem->xfeatMat[j]);
		}

		for (j = 0; j< layer->dim ;j++){
			printf("\n de/dw for node %d\n ",j);
			for (k = 0; k <layer->srcDim;k++){
				printf(" %lf ",layer->info->dwFeatMat[c]);
				c++;
			}

		}
		printf( "\n bias is ");
		for (j = 0; j< layer->dim ;j++){
			printf(" %f ", layer->info->dbFeaMat[j]);
		}

		printf("\n dE/dx for layer %d \n",layer->id);
		for (j = 0; j< layer->srcDim*BATCHSAMPLES;j++){
			printf("%lf ",layer->errElem->dxFeatMat[j]);
		}
	}
}
//-------------------------------------------------------------------------------------------
/*This section deals with running schedulers to iteratively update the parameters of the neural net**/
void addMatrixOrVec(double *weightMat, double* dwFeatMat, int dim){
	//blas routine
	#ifdef CBLAS
		cblas_daxpy(dim,1,weightMat,1,dwFeatMat,1);
	#else
		int i;
		for (i =0;i<dim;i++){
			dwFeatMat[i] = dwFeatMat[i] + weightMat[i];
		}
	#endif	
}

void scaleMatrixOrVec(double* weightMat, double learningrate,int dim){
	//blas routine
	#ifdef CBLAS
		cblas_dscal(dim,learningrate,weightMat,1);
	#else
		int i;
		for (i =0;i<dim;i++){
			weightMat[i] = weightMat[i]*learningrate;	
		}
	#endif	
}
void updateNeuralNetParams(ADLink anndef, double lrnrate, double momentum, double weightdecay){
	int i;
	LELink layer;
	for (i =(anndef->layerNum-1) ; i >=0; --i){
		layer = anndef->layerList[i];
		//if we have a regularised error function: R_E = Er+ b * 1/2 w^w then dR_E = dEr + bw where b is weight decay parameter
		if (weightdecay > 0){
			scaleMatrixOrVec(layer->weights,weightdecay,layer->dim*layer->srcDim);
			scaleMatrixOrVec(layer->bias,weightdecay,layer->dim);
			addMatrixOrVec(layer->weights,layer->info->dwFeatMat,layer->dim*layer->srcDim);
			addMatrixOrVec(layer->bias,layer->info->dbFeaMat,layer->dim);
		}
		if (lrnrate > 0){
			scaleMatrixOrVec(layer->info->dwFeatMat,lrnrate,layer->dim*layer->srcDim);
			scaleMatrixOrVec(layer->info->dbFeaMat,lrnrate,layer->dim);
		}
		if (momentum > 0){
			scaleMatrixOrVec(layer->updateWeightMat,momentum,layer->dim*layer->srcDim);
			scaleMatrixOrVec(layer->updateBiasMat,momentum,layer->dim);
			
			addMatrixOrVec(layer->info->dwFeatMat,layer->updateWeightMat,layer->dim*layer->srcDim);
			addMatrixOrVec(layer->info->dbFeaMat,layer->updateBiasMat,layer->dim);

			//updating parameters: first we need to descale the lambda from weights and bias
			scaleMatrixOrVec(layer->weights,1/weightdecay,layer->dim*layer->srcDim);
			scaleMatrixOrVec(layer->bias,1/weightdecay,layer->dim);

			addMatrixOrVec(layer->updateWeightMat,layer->weights,layer->dim*layer->srcDim);
			addMatrixOrVec(layer->updateBiasMat,layer->bias,layer->dim);
		}else{
			//updating parameters: first we need to descale the lambda from weights and bias
			scaleMatrixOrVec(layer->weights,1/weightdecay,layer->dim*layer->srcDim);
			scaleMatrixOrVec(layer->bias,1/weightdecay,layer->dim);

			addMatrixOrVec(layer->info->dwFeatMat,layer->weights,layer->dim*layer->srcDim);
			addMatrixOrVec(layer->info->dbFeaMat,layer->bias,layer->dim);
		}
	}

}



	



//========================================================================================================
//Note : In the current architecture, I dont specially allocate memory to the ANN to hold labels and input
void freeMemoryfromANN(){
	int i;
	printf("Start freeing memory\n");
	if (anndef != NULL){
		for (i = 0;i<numLayers;i++){
			if (anndef->layerList[i] !=NULL){
				if (anndef->layerList[i]->feaElem != NULL){
					if (anndef->layerList[i]->feaElem->yfeatMat !=NULL){
						free (anndef->layerList[i]->feaElem->yfeatMat);
					}
					free(anndef->layerList[i]->feaElem);
				}
				if (anndef->layerList[i]->errElem !=NULL){
					if (anndef->layerList[i]->errElem->dxFeatMat != NULL){
						free (anndef->layerList[i]->errElem->dxFeatMat);
					}
					free(anndef->layerList[i]->errElem);
				}
				if (anndef->layerList[i]->info!=NULL){
					free (anndef->layerList[i]->info->dwFeatMat);
					free (anndef->layerList[i]->info->dbFeaMat);
					free (anndef->layerList[i]->info);
				}
				if (anndef->layerList[i]->weights !=NULL){
					free (anndef->layerList[i]->weights);
				}
				if (anndef->layerList[i]->bias !=NULL){
					free (anndef->layerList[i]->bias);
				}
				if(anndef->layerList[i]->updateBiasMat !=NULL){
					free(anndef->layerList[i]->updateBiasMat);
				}
				if (anndef->layerList[i]->updateWeightMat!= NULL){
					free(anndef->layerList[i]->updateWeightMat);
				}
				free (anndef->layerList[i]);
			}
		}
		free(anndef->layerList);	
		free(anndef);
	}
	printf("Finished freeing memory\n");	
}

/*This function is used to check the correctness of implementing the forward pass of DNN and the back-propagtion algorithm*/
void unitTests(){
	int i,j,k,c,off;
	LELink layer;
	/*testing forward pass of ANN*
	The structure of ANN is 3 layers : input layer dim =2, hidden layer dim =3, 
	output layer dim =1, the output of ANN is regression
	*/
	//initialise
	ActFunKind list[] = {SIGMOID};
	actfunLists = list;
	targetDim = 2;
	int arr[] ={2};
	hidUnitsPerLayer = arr;
	numLayers = 2; ;
	inputDim = 2;
	target = CLASSIFICATION;
	double lab[] ={0};
	labels = lab;
	printf("before initialisation \n");

	initialiseANN();
	printf("initialisation successful \n");
	double input[] ={1,1};

	anndef->layerList[0]->feaElem->xfeatMat = input;
	anndef->layerList[0]->srcDim = 2;
	
	printf("before forward pass \n");
	fwdPassOfANN();
	printf("forward pass successful \n");
	//------------------------------------------------------------------------------
	//Tests to check forward pass
	/**1st check printin out the weights and bias of the hidden layer **/
	for (i = 0; i <numLayers;i++){
		layer = anndef->layerList[i];
		printf( "\nLayer %d ",i);
		c = 0;
		for (j = 0; j< layer->dim ;j++){
			printf("\n weights for node %d\n ",j);
			for (k = 0; k <layer->srcDim;k++){
				printf("weight is %lf  ",layer->weights[c]);
				c++;
			}

		}
		printf( "\n bias is ");
		for (j = 0; j< layer->dim ;j++){
			printf(" %f ", layer->bias[j]);
		}
		
	}
	/// printing the output of layers
	for (i = 0; i<numLayers;i++){
		layer = anndef->layerList[i];
		printf("\noutput for layer %d\n",i);
		c = 0;
		for (j = 0; j <BATCHSAMPLES;j++){
			printf("\nbatch sample %d\n",j);
			for (k = 0; k<layer->dim;k++){
				printf("%f ",layer->feaElem->yfeatMat[c]);
				c ++;
			}
		}
	}
	//---------------------------------------------------------------------------------------------------------
	/**Comparing backpropagation with symmetrical central differences to check the correctness of the implementation*/
	//Note Run the back-propagation in an on line way that compute the derivative for  a single sample
	double weight_value;
	double biasV;
	weight_value = anndef->layerList[0]->weights[1];
	biasV = anndef->layerList[0]->bias[0];
	anndef->layerList[0]->weights[1]=anndef->layerList[0]->weights[1]+0.0000000001;
	fwdPassOfANN();
	double errorB0 = anndef->layerList[1]->feaElem->yfeatMat[0];
	
	anndef->layerList[0]->weights[1]=anndef->layerList[0]->weights[1]-0.0000000002;
	fwdPassOfANN();
	double errorB20 = anndef->layerList[1]->feaElem->yfeatMat[0];
	
	anndef->layerList[0]->weights[1] = weight_value;
	fwdPassOfANN();

	//testing Back-propagation
	BackPropBatch(anndef);
	printf("\ngradient check  %lf \n",(errorB0-errorB20)/(0.0000000002) );

	freeMemoryfromANN();

}

//=================================================================================

int main(){
	unitTests();
}

