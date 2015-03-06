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
#ifndef CACHESIZE
	#define CACHESIZE 100
#endif
#ifdef CBLAS
#include "/Users/adnan/NeuralNet/CBLAS/include/cblas.h"
#endif

/*hyper-parameters deep Neural Net training initialised with default values*/
static double weightdecay = 0;
static double momentum = 0.4; 
static int  maxEpochNum = 5; 
static double initLR = 0.05; 
static double threshold = 0.5 ; 
static double minLR = 0.0001;

/*training data set and validation data set*/
static int BATCHSAMPLES; //the number of samples to load into the DNN
static double * inputData;
static double * labels;
static double * validationData;
static double * validationLabelIdx;
static int trainingDataSize;
static int validationDataSize;

/*configurations for DNN architecture*/
static MSLink modelSetInfo = NULL;
static ActFunKind *actfunLists;
static int *hidUnitsPerLayer;
static int numLayers;
static int inputDim;
static int targetDim;
static OutFuncKind target;
static ObjFuncKind errfunc;
static ADLink anndef;

//-----------------------------------------------------------------------------------
/**This section of the code parses Command Line arguments**/
//----------------------------------------------------------------------------------
void loadMatrix(double *matrix,){
	FILE *fp;

}





void parseCMDargs(int argc, char *argv[]){
	int i;
	if strcmp(argv[i],"-C")!=0){
		printf("the first argument to ANN must be the config file \n");
		exit(0);
	}
	for (i = 1 ; i <argc;i++){
		if (strcmp(argv[i],"-C")==0){
			++i;
			//parse the config file to set the configurations for DNN architecture
			printf("config file name %s \n",argv[i]);
		}else if(strcmp(argv[i],"-S")==0){
			++i;
			//load the input batch for training
			loadMatrix(inputData);
			printf("training file name %s \n",argv[i]);
		}else if(strcmp(argv[i],"-L")==0){
			++i;
			//load the training labels or outputs in case of regression
			printf("label file name %s \n",argv[i]);
		}else if(strcmp(argv[i],"-v")==0){
			++i;
			//load the validation training samples 
			printf("validation file name %s \n",argv[i]);
		}else if(strcmp(argv[i],"-vL")==0){
			++i;
			//load the validation training labels or expected outputs
			printf("validation label file name %s \n",argv[i]);
		}
		continue;
	}
}




//--------------------------------------------------------------------------------
/**This section of the code deals with handling the batch sizes of the data**/
void setBatchSize(int sampleSize){
	//BATCHSAMPLES = sampleSize < CACHESIZE ? sampleSize : CACHESIZE;
	BATCHSAMPLES = sampleSize;
}

//-----------------------------------------------------------------------------------------------------------
/**this section of the src code deals with initialisation of ANN **/
//-----------------------------------------------------------------------------------------------------------
void reinitLayerMatrices(ADLink anndef){
	LELink layer;
	int i;
	for (i = 0 ; i< anndef->layerNum;i++){
		layer = anndef->layerList[i];
		if (layer->feaElem->yfeatMat != NULL){
			free(layer->feaElem->yfeatMat);
			layer->feaElem->yfeatMat = malloc(sizeof(double)*(layer->dim*BATCHSAMPLES));
			layer->feaElem->xfeatMat = (layer->src != NULL) ? layer->src->feaElem->yfeatMat : NULL;
		}
	}
	for (i = 0;i< anndef->layerNum;i++){
		layer = anndef->layerList[i];
		if (layer->errElem->dxFeatMat != NULL){
			free (layer->errElem->dxFeatMat);
		}
		layer->errElem->dxFeatMat = (double *) malloc(sizeof(double)* (BATCHSAMPLES*layer->srcDim));	
		if ( i!=0){
			layer->src->errElem->dyFeatMat = layer->errElem->dxFeatMat;
		}
	}
}

void initialiseErrElems(ADLink anndef){
	int i;
	LELink layer,srcLayer;
	for (i = 0; i < anndef->layerNum ;i++){
		layer = anndef->layerList[i];
		layer->errElem = (ERLink) malloc (sizeof(ErrElem));
		layer->errElem->dxFeatMat = (double *) malloc(sizeof(double)* (BATCHSAMPLES*layer->srcDim));	
		if ( i!=0){
			srcLayer = layer->src;
			srcLayer->errElem->dyFeatMat = layer->errElem->dxFeatMat;
		}	
	}
	
}
void initialiseWithZero(double *matrix, int dim){
	int i;
	for (i = 0; i< dim;i++){
		*(matrix+i) = 1;
	}
}
 /* uniform distribution, (0..1] */
double drand(){	
return (rand()+1.0)/(RAND_MAX+1.0);
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
			*weights = randm * 1/(sqrt(srcDim));
			weights = weights + 1;
		}
	}
}
void initialiseLayer(LELink layer,int i, LELink srcLayer){
	int srcDim,numOfElems;
	if (srcLayer != NULL ) srcDim = srcLayer->dim;
	else srcDim = inputDim;
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
	assert(layer->info!= NULL);
	layer->info->dwFeatMat = malloc(sizeof(double)*numOfElems);
	layer->info->dbFeaMat = malloc(sizeof(double)*layer->dim);
	layer->info->updateWeightMat = NULL;
	layer->info->updateBiasMat = NULL;

	if (momentum > 0) {
		layer->info->updateWeightMat = malloc(sizeof(double)*numOfElems);
		layer->info->updateBiasMat = malloc(sizeof(double)*(layer->dim));
		initialiseWithZero(layer->info->updateWeightMat,numOfElems);
		initialiseWithZero(layer->info->updateBiasMat,layer->dim);
	}
}

void  initialiseDNN(){
	int i;
	anndef = malloc(sizeof(ANNDef));
	assert(anndef!=NULL);
	anndef->target = target;
	anndef->layerNum = numLayers;
	anndef->layerList = (LELink *) malloc (sizeof(LELink)*numLayers);
	assert(anndef->layerList!=NULL);
	/*initialise the seed only once then initialise weights and bias*/
	srand((unsigned int)time(NULL));
	//initilaise layers
	for(i = 0; i<anndef->layerNum; i++){
		anndef->layerList[i] = (LELink) malloc (sizeof(LayerElem));
		assert (anndef->layerList[i]!=NULL);
		if (i == 0 ){
			initialiseLayer(anndef->layerList[i],i, NULL);
		}else{
			initialiseLayer(anndef->layerList[i],i, anndef->layerList[i-1]);
		}	
	}
	anndef->errorfunc = errfunc;
	//initialise ErrElems of layers for back-propagation
	initialiseErrElems(anndef);
}
//------------------------------------------------------------------------------------------------
/*this section of the code implements the  forward propgation of a deep neural net **/
//-------------------------------------------------------------------------------------------------
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
void fwdPassOfANN(ADLink anndef){
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
//------------------------------------------------------------------------------------------------------
/*This section of the code implements the back-propation algorithm  to compute the error derivatives*/
//-------------------------------------------------------------------------------------------------------

void computeDrvAct(double *dyfeat , double *yfeat,int len){
	//CPU Version
	int i;
	for (i = 0; i< len;i++){
		dyfeat[i] = dyfeat[i]*yfeat[i];
	}
}

void computeActivationDrv (LELink layer){
	int i;
	switch (layer->actfuncKind){
		case SIGMOID:
			//CPU verion
		  for (i = 0; i<layer->dim*BATCHSAMPLES;i++){
				layer->feaElem->yfeatMat[i] = layer->feaElem->yfeatMat[i]*(1-layer->feaElem->yfeatMat[i]);
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
	#else
	//CPU version
		int i;
		for (i = 0; i<dim;i++){
			dyfeat[i] = dyfeat[i]-labels[i];
		}
	#endif
}

void CalcOutLayerBackwardSignal(LELink layer,ADLink anndef ){
	switch(anndef->errorfunc){
		case (XENT):
			switch(layer->actfuncKind){
				case SIGMOID:
					subtractMatrix(layer->errElem->dyFeatMat,anndef->labelMat,layer->dim*BATCHSAMPLES);
				break;
				case SOFTMAX:
					subtractMatrix(layer->errElem->dyFeatMat,anndef->labelMat,layer->dim*BATCHSAMPLES);
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
		layer = anndef->layerList[i];
		if (layer->role ==OUTPUT){
			layer->errElem->dyFeatMat = layer->feaElem->yfeatMat;
			CalcOutLayerBackwardSignal(layer,anndef);
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
	}
}	


//----------------------------------------------------------------------------------------------------
/*This section deals with running schedulers to iteratively update the parameters of the neural net**/
//-------------------------------------------------------------------------------------------------------
void perfBinClassf(double *yfeatMat, double *predictions,int dataSize){
	int i;
	for (i = 0; i< dataSize;i++){
		predictions[i] = yfeatMat[i]>0.5 ? 1 :0;
		printf("Predictions %d  %lf  and yfeat is %lf \n",i,predictions[i],yfeatMat[i]);
	}
}

/*The function finds the most active node in the output layer for each sample*/
void findMaxElement(double *matrix, int row, int col, double *vec){
	int maxIdx, i, j;
   double maxVal;
	for (i = 0; i < row; ++i) {
      maxIdx = 0;
      maxVal = matrix[i * col + 0];
      for (j = 1; j < col; ++j) {
         if (maxVal < matrix[i * col + j]) {
            maxIdx = j;
            maxVal = matrix[i * col + j];
            }
        }
      vec[i] = maxIdx;
   }
}
/** the function calculates the percentage of the data samples correctly labelled by the DNN*/
void updatateAcc(double *labels, LELink layer,int dataSize){
	int i, dim, accCount;
	double *predictions = malloc(sizeof(double)*dataSize);
	if (layer->dim >1){
		dim = layer->dim;
		findMaxElement(layer->feaElem->yfeatMat,dataSize,dim,predictions);
	}else{
		perfBinClassf(layer->feaElem->yfeatMat,predictions,dataSize);
	}
	accCount = 0;
	for (i = 0; i<dataSize;i++){
		if (predictions[i] == labels[i]){
			accCount+=1;
		}	
	}
	free(predictions);
	modelSetInfo->crtVal = ((double)accCount)/((double) dataSize);
	printf("The critical value is %f \n", modelSetInfo->crtVal);
}

/* this function allows the addition of  two matrices or two vectors*/
void addMatrixOrVec(double *dwFeatMat, double* weights, int dim){
	//blas routine
	#ifdef CBLAS
		cblas_daxpy(dim,1,dwFeatMat,1,weights,1);
	#else
		int i;
		for (i =0;i<dim;i++){
			weights[i] = dwFeatMat[i] + weightMat[i];
		}
	#endif	
}
/*multipy a vector or a matrix with a scalar*/
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
			scaleMatrixOrVec(layer->info->updateWeightMat,momentum,layer->dim*layer->srcDim);
			scaleMatrixOrVec(layer->info->updateBiasMat,momentum,layer->dim);
			addMatrixOrVec(layer->info->dwFeatMat,layer->info->updateWeightMat,layer->dim*layer->srcDim);
			addMatrixOrVec(layer->info->dbFeaMat,layer->info->updateBiasMat,layer->dim);
			//updating parameters: first we need to descale the lambda from weights and bias
			if (weightdecay > 0){
			scaleMatrixOrVec(layer->weights,1/weightdecay,layer->dim*layer->srcDim);
			scaleMatrixOrVec(layer->bias,1/weightdecay,layer->dim);
			}
			addMatrixOrVec(layer->info->updateWeightMat,layer->weights,layer->dim*layer->srcDim);
			addMatrixOrVec(layer->info->updateBiasMat,layer->bias,layer->dim);
		}else{
			//updating parameters: first we need to descale the lambda from weights and bias
			if (weightdecay > 0){
				scaleMatrixOrVec(layer->weights,1/weightdecay,layer->dim*layer->srcDim);
				scaleMatrixOrVec(layer->bias,1/weightdecay,layer->dim);
			}
			addMatrixOrVec(layer->info->dwFeatMat,layer->weights,layer->dim*layer->srcDim);
			addMatrixOrVec(layer->info->dbFeaMat,layer->bias,layer->dim);
		}
	}
}

void updateLearningRate(int currentEpochIdx, double *lrnrate){
	double crtvaldiff;
	if (currentEpochIdx == 0) {
		*lrnrate = (-1)* initLR;
	}else if (modelSetInfo !=NULL){
		crtvaldiff = modelSetInfo->crtVal - modelSetInfo->prevCrtVal;
		if (crtvaldiff < threshold){
			*lrnrate /=2;
			printf("Learning rate has been halved !! \n");
		}
	}
}

Boolean terminateSchedNotTrue(int currentEpochIdx,double lrnrate){
	printf("lrn rate %f\n",lrnrate);
	if (currentEpochIdx == 0) return TRUE;
	if (currentEpochIdx >=0 && currentEpochIdx >= maxEpochNum)return FALSE;
	lrnrate *=-1;
	if( lrnrate < minLR)return FALSE;
		
	
	return TRUE; 
}

void TrainDNN(){
	int currentEpochIdx;
	double learningrate;
	
	currentEpochIdx = 0;
	learningrate = 0;
	modelSetInfo = malloc (sizeof(MSI));
	modelSetInfo->crtVal = 0;

	printf("initialising DNN\n");
	initialiseDNN();
	printf("successfully initialised DNN\n");
	
	//with the initialisation of weights,check how well DNN performs on validation data
	setBatchSize(validationDataSize);
	reinitLayerMatrices(anndef);
	anndef->layerList[0]->feaElem->xfeatMat = validationData;
	anndef->labelMat = validationLabelIdx;
	
	fwdPassOfANN(anndef);
	printf("successfully performed forward pass of DNN on validation data\n");
	updatateAcc(validationLabelIdx, anndef->layerList[numLayers-1],validationDataSize);
	
	int count = 0;
	while(terminateSchedNotTrue(currentEpochIdx,learningrate)){
		count+=1;
		printf("epoc number %d \n", currentEpochIdx);
		updateLearningRate(currentEpochIdx,&learningrate);
		//load training data into the ANN and perform forward pass
		setBatchSize(trainingDataSize);
		reinitLayerMatrices(anndef);
		anndef->layerList[0]->feaElem->xfeatMat = inputData;
		//BATCHSAMPLES =  trainingDataSize;
		anndef->labelMat = labels ;
		fwdPassOfANN(anndef);
		
		// run backpropagation and update the parameters:
		BackPropBatch(anndef);
		updateNeuralNetParams(anndef,learningrate,momentum,weightdecay);
		
		//forward pass of DNN on validation data if VD is provided
		if (validationData != NULL && validationLabelIdx != NULL){
			setBatchSize(validationDataSize);
			reinitLayerMatrices(anndef);
			anndef->layerList[0]->feaElem->xfeatMat = validationData;
			anndef->labelMat = validationLabelIdx;
			//perform forward pass on validation data and check the performance of the DNN on the validation dat set
			fwdPassOfANN(anndef);
			modelSetInfo->prevCrtVal = modelSetInfo->crtVal;
			updatateAcc(validationLabelIdx,anndef->layerList[numLayers-1],validationDataSize);
		}else{
			modelSetInfo->prevCrtVal = modelSetInfo->crtVal;
			updatateAcc(labels,anndef->layerList[numLayers-1],trainingDataSize);
		}
		currentEpochIdx+=1;
	}
	printf("COUNT is %d \n",count);




}
//========================================================================================================
//Note : In the current architecture, I dont specially allocate memory to the ANN to hold labels and input
void freeMemoryfromANN(){
	int i;
	printf("Start freeing memory\n");
	if (anndef != NULL){
		for (i = 0;i<numLayers;i++){
			printf("LAYER %d \n",i);
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
					if(anndef->layerList[i]->info->updateBiasMat !=NULL){
						free(anndef->layerList[i]->info->updateBiasMat);
					}
					if (anndef->layerList[i]->info->updateWeightMat!= NULL){
						free(anndef->layerList[i]->info->updateWeightMat);
					}
					free (anndef->layerList[i]->info);
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
		free(modelSetInfo);
	}
	printf("Finished freeing memory\n");	
}

/*This function is used to check the correctness of implementing the forward pass of DNN and the back-propagtion algorithm*/
void unitTests(){
	ActFunKind list[] = {SIGMOID,SIGMOID, SOFTMAX};
	actfunLists = list;
	targetDim = 2;
	
	int arr[] ={5,4};
	hidUnitsPerLayer = arr;
	
	numLayers = 3; ;
	
	
	target = CLASSIFICATION;
	double lab[] ={0,1,0,1};
	labels = lab;
	double lab2[] ={0,1,0,1};
	validationLabelIdx = lab2;

	double input[] = { 1,3,2,4,3,5,6,8};
	trainingDataSize = 4;
	setBatchSize(4);
	inputDim = 2;
	inputData =input;

	double test[] = { 5,3,6,8,7,1,10,8};
	validationDataSize = 4;
	validationData = test ;

	
	TrainDNN();


	/**
	int i,j,k,c,off;
	LELink layer;
	testing forward pass of ANN*
	The structure of ANN is 3 layers : input layer dim =2, hidden layer dim =3, 
	output layer dim =1, the output of ANN is regression
	
	//initialise
	ActFunKind list[] = {SIGMOID};
	actfunLists = list;
	targetDim = 2;
	int arr[] ={2};
	hidUnitsPerLayer = arr;
	numLayers = 2; ;
	inputDim = 2;
	target = CLASSIFICATION;
	int lab[] ={0};
	labels = lab;
	printf("before initialisation \n");

	initialiseDNN();
	printf("initialisation successful \n");
	double input[] ={1,1};

	anndef->layerList[0]->feaElem->xfeatMat = input;
	anndef->layerList[0]->srcDim = 2;
	
	printf("before forward pass \n");
	fwdPassOfANN(anndef);
	printf("forward pass successful \n");
	//------------------------------------------------------------------------------
	**///Tests to check forward pass
	//1st check printin out the weights and bias of the hidden layer 
	/**for (i = 0; i <numLayers;i++){
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
	//Comparing backpropagation with symmetrical central differences to check the correctness of the implementation
	//Note Run the back-propagation in an on line way that compute the derivative for  a single sample
	double weight_value;
	double biasV;
	weight_value = anndef->layerList[0]->weights[1];
	biasV = anndef->layerList[0]->bias[0];
	anndef->layerList[0]->weights[1]=anndef->layerList[0]->weights[1]+0.0000000001;
	fwdPassOfANN(anndef);
	double errorB0 = anndef->layerList[1]->feaElem->yfeatMat[0];
	
	anndef->layerList[0]->weights[1]=anndef->layerList[0]->weights[1]-0.0000000002;
	fwdPassOfANN(anndef);
	double errorB20 = anndef->layerList[1]->feaElem->yfeatMat[0];
	
	anndef->layerList[0]->weights[1] = weight_value;
	fwdPassOfANN(anndef);

	//testing Back-propagation
	BackPropBatch(anndef);
	printf("\ngradient check  %lf \n",(errorB0-errorB20)/(0.0000000002) );
	**/
	/*testing to check the correctness of implementation of learning rate schedulars*/
	//initialise



	freeMemoryfromANN();

}


//=================================================================================

int main(int argc, char *argv[]){
	if (argc != 11 && argc != 12 ){
		printf("The program expects a minimum of  5 args and a maximum of 6 args : Eg : -C config \n -S traindatafile \n -L traininglabels \n -v validationdata \n -vl validationdataLabels \n optional argument : -T testData \n ");
	}
	parseCMDargs(argc, argv);

FILE *fp;
size_t len = 0;
char *line =NULL;
fp = fopen("/Users/adnan/NeuralNet/NeuralNet/tmp","r");	
char* token;
double value;
while (getline(&line, &len, fp)!= -1){
	token = strtok(line, " ");
	while (token != NULL){
		value = strtod(token ,NULL);
		printf( "%lf",value);
		token = strtok(NULL," ");
	}
	printf("\n");
}



	//unitTests();
}

