#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "ANN.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

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
static  int maxNumOfCGruns = 10;

/*training data set and validation data set*/
static int BATCHSAMPLES; //the number of samples to load into the DNN
static double * inputData;
static double * labels;
static double * validationData;
static double * validationLabelIdx;
static int trainingDataSetSize;
static int validationDataSetSize;


/*configurations for DNN architecture*/
static Boolean doHF = FALSE;
static Boolean useGNMatrix = FALSE;
static MSLink modelSetInfo = NULL;
static ActFunKind *actfunLists;
static int *hidUnitsPerLayer;
static int numLayers;
static int inputDim;
static int targetDim;
static OutFuncKind target;
static ObjFuncKind errfunc; 
static ADLink anndef = NULL;

//-----------------------------------------------------------------------------------
/**This section of the code deals with parsing Command Line arguments**/
//----------------------------------------------------------------------------------
void cleanString(char *Name){
	char *pos;
	if ((pos=strchr(Name, '\n')) != NULL)
	    *pos = '\0';
}
void loadLabels(double *labelMat,char*filepath,char *datatype){
	FILE *fp;
	int i,c;
	char *line = NULL;
	size_t len = 0;
	int id ;
	int samples = 0;
	samples  = strcmp(datatype,"train")== 0 ? trainingDataSetSize: validationDataSetSize;

	fp = fopen(filepath,"r");
	i = 0;
	while(getline(&line,&len,fp)!=-1){
		cleanString(line);
		//extracting labels 
		if (strcmp(datatype,"train")==0){
			id  = strtod(line,NULL);
			labelMat[i*targetDim+id] = 1;
			if (i> trainingDataSetSize){
				printf("Error! : the number of training labels doesnt match the size of the training set \n");
				exit(0);
			}
		}else if(strcmp(datatype,"validation")==0){
			id  = strtod(line,NULL);
			labelMat[i] = id;
			if(i > validationDataSetSize){
				printf("Error! : the number of validation target labels doesnt match the size of the validation set \n");
				exit(0);
			}
		}
		i+=1;
	}
	free(line);
	fclose(fp);		
}

void loadMatrix(double *matrix,char *filepath, char *datatype){
	FILE *fp;
	int i;
	char *line = NULL;
	size_t len = 0;
	char* token;
	
	fp = fopen(filepath,"r");
	i = 0;
	while(getline(&line,&len,fp)!=-1){
		token = strtok(line,",");
		while (token != NULL){
			matrix[i] = strtod(token,NULL);
			token = strtok(NULL,",");
			if (strcmp(datatype,"train")== 0){
				if (i > trainingDataSetSize*inputDim){
					printf("Error: either the size of the training set or the dim of the  feature vectors have been incorrectly specified in config file \n");
					exit(0);
				}
			}else if (strcmp(datatype,"validation")==0){
				if (i > validationDataSetSize*inputDim){
					printf("Error: either the size of the   validation dataset or the dim of the  target vectors have been incorrectly specified in config file \n");
					exit(0);
				}
			}
			i+=1;
		}
	}	
	free(line);
	fclose(fp);
}

void parseCfg(char * filepath){
	FILE *fp;
	char *line = NULL;
	size_t len = 0;
	char* token;
	char* list;
	char *pos;

	fp = fopen(filepath,"r");
	while(getline(&line,&len,fp)!=-1){
		token = strtok(line," : ");
		if (strcmp(token,"momentum")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			momentum = strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"weightdecay")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			weightdecay = strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;	
		}
		if (strcmp(token,"minLR")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			minLR = strtod (token,NULL);
			token = strtok(NULL," : ");
			continue	;
		}
		if (strcmp(token,"maxEpochNum")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			maxEpochNum = (int) strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"initLR")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			initLR =  strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"threshold")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			threshold = strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"numLayers")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			numLayers = (int) strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"inputDim")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			inputDim = (int) strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"targetDim")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			targetDim = (int) strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"trainingDataSetSize")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			trainingDataSetSize = (int) strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"validationDataSetSize")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			validationDataSetSize = (int)strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"Errfunc")==0){
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			if (strcmp("XENT",token)==0){
				errfunc = XENT;
				target = CLASSIFICATION;
			}else if (strcmp("SSE",token)==0){
				printf("ITS Not XENT\n");
				errfunc = SSE ;
				target = REGRESSION ;
			}	
			continue;
		}
		
		if (strcmp(token,"hiddenUnitsPerLayer")==0){
			hidUnitsPerLayer = malloc(sizeof(int)*numLayers);
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			list = token;
			while(token != NULL){
				token = strtok(NULL,":");
			}	
			token = strtok(list,",");
			cleanString(token);
			int count = 0;
			while(token !=NULL){
				*(hidUnitsPerLayer+count) = (int) strtod (token,NULL);
				count+=1;
				token = strtok(NULL,",");
				if (token == NULL) break;
				cleanString(token);
			}
			continue;
		}
		if (strcmp(token,"activationfunctionsPerLayer")==0){
			actfunLists = malloc( sizeof(ActFunKind)*numLayers);
			token = strtok(NULL," : ");
			if ((pos=strchr(token, '\n')) != NULL){*pos = '\0';}
			list = token;
			while(token !=NULL){
				token = strtok(NULL,":");
			}
			token = strtok(list,",");
			cleanString(token);
		   int count = 0;
			while(token !=NULL){
			if (strcmp(token,"SIGMOID")==0){
				*(actfunLists+count) = SIGMOID ;
				}else if (strcmp(token,"IDENTITY")==0){
					*(actfunLists+count) = IDENTITY;
				}else if (strcmp(token,"TANH")==0){
					*(actfunLists+count) = TANH ;
				}else if (strcmp(token,"SOFTMAX")==0){
					*(actfunLists+count) = SOFTMAX;
			}		
				count+=1;
				token = strtok(NULL,",");
				if (token ==NULL) break;
				cleanString(token);
			}	
			continue;
		}
		
	}
	free(line);
	fclose(fp);
}

void parseCMDargs(int argc, char *argv[]){
	int i;
	if (strcmp(argv[1],"-C")!=0){
		printf("the first argument to ANN must be the config file \n");
		exit(0);
	}
	for (i = 1 ; i < argc;i++){
		if (strcmp(argv[i],"-C") == 0){
			++i;
			printf("parsing cfg\n");
			parseCfg(argv[i]);
			//parse the config file to set the configurations for DNN architecture
			printf("config file %s has been successfully parsed \n",argv[i]);
			continue;
		}
	   if(strcmp(argv[i],"-S") == 0){
	   	++i;
			//load the input batch for training
			printf("parsing training data file \n");
			inputData = malloc(sizeof(double)*(trainingDataSetSize*inputDim));
			loadMatrix(inputData,argv[i],"train");
			printf("training samples from %s have been successfully loaded \n",argv[i]);
			continue;
		}
		if(strcmp(argv[i],"-L")==0){
			++i;
			//load the training labels or outputs in case of regression
			printf("parsing training-labels file \n");
			labels = malloc(sizeof(double)*(trainingDataSetSize*targetDim));
			initialiseWithZero(labels,trainingDataSetSize*targetDim);
			loadLabels(labels,argv[i],"train");
			printf("training labels from %s have been successfully loaded \n",argv[i]);
			continue;
		} 
		if(strcmp(argv[i],"-v")==0){
			++i;
			//load the validation training samples 
			printf("parsing validation-data file \n");
			validationData = malloc (sizeof(double)*(validationDataSetSize*inputDim));
			loadMatrix(validationData,argv[i],"validation");
			printf("samples from validation file %s have been successfully loaded \n",argv[i]);
			continue;
		}
		if(strcmp(argv[i],"-vl")==0){
			++i;
			//load the validation training labels or expected outputs
			printf("parsing validation-data-label file \n");
			validationLabelIdx = malloc(sizeof(double)*(validationDataSetSize));
			initialiseWithZero(validationLabelIdx,validationDataSetSize);
			loadLabels(validationLabelIdx,argv[i],"validation");
			printf("validation labels from %s have been successfully loaded\n",argv[i]);
			continue;
		}
	}
}

//--------------------------------------------------------------------------------
/**This section of the code deals with handling the batch sizes of the data**/
void setBatchSize(int sampleSize){
	//BATCHSAMPLES = sampleSize < CACHESIZE ? sampleSize : CACHESIZE;
	BATCHSAMPLES = sampleSize;
}
/**load entire batch into the neural net**/
void loadDataintoANN(double *samples, double *labels){
	anndef->layerList[0]->feaElem->xfeatMat = samples;
	anndef->labelMat = labels;
}   


//-----------------------------------------------------------------------------------------------------------
/**this section of the src code deals with initialisation of ANN **/
//-----------------------------------------------------------------------------------------------------------
void setUpForHF(ADLink anndef){
	LELink layer;
	int i,srdim,dim;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		//set up structure to accumulate gradients
		if (layer->traininfo->updatedWeightMat == NULL && layer->traininfo->updatedBiasMat == NULL){
			layer->traininfo->updatedWeightMat = malloc(sizeof(double)*(layer->dim * layer->srcDim));
			layer->traininfo->updatedBiasMat = malloc(sizeof(double)*(layer->dim));
			initialiseWithZero(layer->traininfo->updatedWeightMat,layer->dim*layer->srcDim);
			initialiseWithZero(layer->traininfo->updatedBiasMat,layer->dim);
		}
		else if (layer->traininfo->updatedWeightMat == NULL || layer->traininfo->updatedBiasMat == NULL){
			printf("Error something went wrong during the initialisation of updatedWeightMat and updateBiasMat in the layer %d \n",i);
			exit(0);
		}
		layer->cgInfo = malloc(sizeof(ConjuageGradientInfo));
		layer->cgInfo->delweightsUpdate = malloc(sizeof(double)*(layer->dim*layer->srcDim));
		layer->cgInfo->residueUpdateWeights = malloc(sizeof(double)*(layer->dim*layer->srcDim));
		layer->cgInfo->searchDirectionUpdateWeights = malloc(sizeof(double)*(layer->dim*layer->srcDim));
		initialiseWithZero(layer->cgInfo->delweightsUpdate,layer->dim*layer->srcDim);
		initialiseWithZero(layer->cgInfo->residueUpdateWeights,layer->dim*layer->srcDim);
		initialiseWithZero(layer->cgInfo->searchDirectionUpdateWeights,layer->dim*layer->srcDim);

		layer->cgInfo->delbiasUpdate = malloc(sizeof(double)*layer->dim);
		layer->cgInfo->searchDirectionUpdateBias = malloc(sizeof(double)*layer->dim);
		layer->cgInfo->residueUpdateBias = malloc(sizeof(double)*layer->dim);
		initialiseWithZero(layer->cgInfo->delbiasUpdate,layer->dim);
		initialiseWithZero(layer->cgInfo->residueUpdateBias,layer->dim);
		initialiseWithZero(layer->cgInfo->searchDirectionUpdateBias,layer->dim);


		if (useGNMatrix){
			layer->gnInfo = malloc (sizeof(GaussNewtonProductInfo));
			layer->gnInfo->vweights = malloc(sizeof(double)*(layer->dim*layer->srcDim));
			layer->gnInfo->vbiases = malloc(sizeof(double)* layer->dim);
			layer->gnInfo->Ractivations  = malloc(sizeof(double)*(layer->dim*BATCHSAMPLES));
		}
	}
}

void reinitLayerFeaMatrices(ADLink anndef){
	LELink layer;
	int i;
	for (i = 0 ; i< anndef->layerNum;i++){
		layer = anndef->layerList[i];
		if (layer->feaElem->yfeatMat != NULL){
			free(layer->feaElem->yfeatMat);
			layer->feaElem->yfeatMat = malloc(sizeof(double)*(layer->dim*BATCHSAMPLES));
			initialiseWithZero(layer->feaElem->yfeatMat,layer->dim*BATCHSAMPLES);
			layer->feaElem->xfeatMat = (layer->src != NULL) ? layer->src->feaElem->yfeatMat : NULL;
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
		*(matrix+i) = 0;
	}
}
double genrandWeight(double limit){
	return  -limit + (2*limit)*drand()  ;
}

 /* uniform distribution, (0..1] */
double drand(){	
return (double) rand()/(RAND_MAX);
}

/* performing the Box Muller transform to map two numbers 
generated from a uniform distribution to a number from a normal distribution centered at 0 with standard deviation 1 */
double random_normal() {
  return sqrt(-2*log(drand())) * cos(2*M_PI*drand());
}

void initialiseBias(double *biasVec,int dim, int srcDim ,ActFunKind actfunc){
	int i;
	double randm;
	for ( i = 0; i<dim;i++){
		/* bengio;s proposal for a new type of initialisation to ensure 
			the variance of error derivatives are comparable accross layers*/
		if (actfunc==SIGMOID){
			biasVec[i] = 4 *genrandWeight(sqrt(6)/sqrt(dim+srcDim));
		}else{
			biasVec[i] = genrandWeight(sqrt(6)/sqrt(dim+srcDim));
		}
	}
}
/*the srcDim determines the fan-in to the hidden and output units. The weights ae initialised 
to be inversely proportional to the sqrt(fanin)
*/ 
void initialiseWeights(double *weights,int dim,int srcDim, ActFunKind actfunc){
	int i,j;
	double randm;
	//this is not an efficient way of doing but it allows better readibility
	for (i = 0; i < dim; i++){
		for(j = 0; j < srcDim;j++){
			/* bengio;s proposal for a new tpye of initialisation to ensure 
			the variance of error derivatives are comparable accross layers*/
			if (actfunc == SIGMOID){
				*weights = 4* genrandWeight(sqrt(6)/sqrt(dim+srcDim+1));
			}else{
				*weights = genrandWeight(sqrt(6)/sqrt(dim+srcDim+1));
			}
			weights = weights + 1;
		}
	}
}
void initialiseLayer(LELink layer,int i, LELink srcLayer){
	int srcDim,numOfElems;
	if (srcLayer != NULL ) {
		srcDim = srcLayer->dim;
	}
	else {
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
	layer->bias = malloc(sizeof(double)*(layer->dim));
	assert(layer->bias!=NULL);
	//initialise weights of outer layer
	if (i ==(numLayers-1)){
		initialiseWithZero(layer->bias,layer->dim);
		initialiseWithZero(layer->weights,layer->dim * layer->srcDim);
	}else{
		//initialise weights of hidden layers
		initialiseWeights(layer->weights,layer->dim,layer->srcDim,layer->actfuncKind);
		//initialiseBias(layer->bias,layer->dim,layer->srcDim,layer->actfuncKind);
		initialiseWithZero(layer->bias,layer->dim);
	}
	
	//initialise feaElems
	layer->feaElem = (FELink) malloc(sizeof(FeaElem));
	assert(layer->feaElem!=NULL);
	layer->feaElem->yfeatMat = malloc(sizeof(double)*(layer->dim*BATCHSAMPLES));
	layer->feaElem->xfeatMat = (srcLayer != NULL) ? srcLayer->feaElem->yfeatMat : NULL;
	
	//intialise traininfo and allocating extra memory for setting hooks
	layer->traininfo = (TRLink) malloc(sizeof(TrainInfo) * sizeof(double)*(numOfElems*4));
	assert(layer->traininfo!= NULL);
	layer->traininfo->dwFeatMat = malloc(sizeof(double)*numOfElems);
	layer->traininfo->dbFeaMat = malloc(sizeof(double)*layer->dim);
	layer->traininfo->updatedWeightMat = NULL;
	layer->traininfo->updatedBiasMat = NULL;

	if (momentum > 0) {
		layer->traininfo->updatedWeightMat = malloc(sizeof(double)*numOfElems);
		layer->traininfo->updatedBiasMat = malloc(sizeof(double)*(layer->dim));
		initialiseWithZero(layer->traininfo->updatedWeightMat,numOfElems);
		initialiseWithZero(layer->traininfo->updatedBiasMat,layer->dim);
	}
}

void initialiseDNN(){
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
	if (doHF) {
		setUpForHF(anndef);
	}

}
void initialise(){
	printf("initialising DNN\n");
	setBatchSize(trainingDataSetSize);
	initialiseDNN();
	printf("successfully initialised DNN\n");
}
//------------------------------------------------------------------------------------------------
/*this section of the code implements the  forward propgation of a deep neural net **/
//-------------------------------------------------------------------------------------------------
void copyMatrixOrVec(double *src, double *dest,int dim){
	#ifdef CBLAS
	cblas_dcopy(dim, src, 1,dest, 1);
	#else
	memcpy(dest,src,sizeof(double)*dim);		
	#endif
}
/* this function allows the addition of  two matrices or two vectors*/
void addMatrixOrVec(double *weights, double *dwFeatMat,int dim, double lambda){
	//blas routine
	#ifdef CBLAS
		cblas_daxpy(dim,lambda,weights,1,dwFeatMat,1);
	#else
		int i;
		for (i =0;i<dim;i++){
			dwFeatMat[i] = dwFeatMat[i] + lambda*weights[i];
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
void subtractMatrix(double *dyfeat, double* labels, int dim){
	//blas routine
	#ifdef CBLAS
		cblas_daxpy(dim,-1.0,labels,1,dyfeat,1);
	#else
	//CPU version
		int i;
		for (i = 0; i<dim;i++){
			dyfeat[i] = dyfeat[i]-labels[i];
		}
	#endif
}

double computeTanh(double x){
	return 2*(computeSigmoid(2*x))-1;
}
double computeSigmoid(double x){
	double result;
	result = 1/(1+ exp(-x));
	return result;
}
/*computing non-linear activation*/
void computeNonLinearActOfLayer(LELink layer){
	int i,j ;
	double sum;
	switch(layer->role){
		case HIDDEN:
			switch(layer->actfuncKind){
				case SIGMOID:
				for (i = 0;i < layer->dim*BATCHSAMPLES;i++){
				//	printf("result before %lf \n ",layer->feaElem->yfeatMat[i]);
					layer->feaElem->yfeatMat[i] = computeSigmoid(layer->feaElem->yfeatMat[i]);
					//printf("result %lf \n ",layer->feaElem->yfeatMat[i]);
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
					for (i = 0;i < BATCHSAMPLES;i++){
						sum = 0;
						for (j = 0; j<layer->dim;j++){
							double value = layer->feaElem->yfeatMat[i*layer->dim+j];
							layer->feaElem->yfeatMat[i*layer->dim+j] = exp(value);
							sum= sum+ exp(value);
						}
						for (j =0; j<layer->dim;j++){
							layer->feaElem->yfeatMat[i*layer->dim+j]= layer->feaElem->yfeatMat[i*layer->dim+j]/sum ;
						}
					}
					break;
				default:
					break;	
			}
			break;
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
					computeNonLinearActOfLayer(layer);
				}
				computeLinearActivation(anndef->layerList[anndef->layerNum-1]);
				break;
			case CLASSIFICATION:
				for (i = 0; i< anndef->layerNum;i++){
					layer = anndef->layerList[i];
					computeLinearActivation(layer);
					computeNonLinearActOfLayer(layer);
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
			break;
		case TANH:
			//CPU verion
		  for (i = 0; i<layer->dim*BATCHSAMPLES;i++){
				layer->feaElem->yfeatMat[i] = 4*layer->feaElem->yfeatMat[i]*(1-layer->feaElem->yfeatMat[i]);
			}
			break;
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
		free (ones);
	#endif
}
/**compute del^2L J where del^2L is the hessian of the cross-entropy softmax with respect to output acivations **/ 

void computeLossHessSoftMax(LELink layer){
	int i,j;
	double *RactivationVec = malloc(sizeof(double)*layer->dim);
	double *yfeatVec = malloc(sizeof(double)*layer->dim);
	double *diaP = malloc(sizeof(double)*layer->dim*layer->dim);
	double *result = malloc(sizeof(double)*layer->dim);
	
	for (i = 0 ; i< BATCHSAMPLES; i++){
		#ifdef CBLAS
		/**extract error directional derivative for a single sample*/ 
		cblas_dcopy(layer->dim, layer->gnInfo->Ractivations+i*(layer->dim), 1, RactivationVec, 1);
		cblas_dcopy(layer->dim,layer->feaElem->yfeatMat+i*(layer->dim), 1, yfeatVec, 1);

		//compute dia(yfeaVec - yfeacVec*yfeaVec'
		cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,layer->dim,layer->dim,1,-1,yfeatVec,layer->dim,yfeatVec,1,0,diaP,layer->dim);
		for (j = 0; j<layer->dim;j++){
			diaP[j*(layer->dim+1)] += yfeatVec[j];
		}
		//multiple hessian of loss function of particular sample with Jacobian 
		cblas_dgemv(CblasColMajor,CblasNoTrans,layer->dim,layer->dim,1,diaP,layer->dim,RactivationVec,1,0,result,1);
		cblas_dcopy(layer->dim, result, 1,layer->gnInfo->Ractivations+i*(layer->dim),1);
		#else
			RactivationVec = memcpy(RactivationVec,layer->gnInfo->Ractivations+i, sizeof(double)*layer->dim);
			yfeatVec = memcpy(yfeatVec,layer->feaElem->yfeatMat+i,sizeof(double)*layer->dim);
		#endif
	}
	free(result);
	free(yfeatVec);
	free(RactivationVec);
	free(diaP);

}

/*compute del^2L*J where L can be any convex loss function**/
void computeHessOfLossFunc(LELink outlayer, ADLink anndef){
	switch(anndef->errorfunc){
		case (XENT):
			switch (outlayer->actfuncKind){
				case SIGMOID:
					//for each sample del2 Loss = diag(P*(1-p))where P is Prediction so we multiply diag(P) with J
					computeActivationDrv (outlayer);
					computeDrvAct(outlayer->gnInfo->Ractivations, outlayer->feaElem->yfeatMat,outlayer->dim*BATCHSAMPLES);
					break;
				case SOFTMAX:
					computeLossHessSoftMax(outlayer);
				default:
					break;
			}
		case SSE :
			break;
	}
}

void calcOutLayerBackwardSignal(LELink layer,ADLink anndef ){
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
				default:
					break;
			}
		break;	
		case (SSE):
			subtractMatrix(layer->errElem->dyFeatMat,anndef->labelMat,layer->dim*BATCHSAMPLES);	
			break;
	}
}

/**function computes the error derivatives with respect to the weights and biases of the neural net*/
void backPropBatch(ADLink anndef,Boolean doHessVecProd){
	int i;
	LELink layer;
	for (i = (anndef->layerNum-1); i>=0;i--){
		layer = anndef->layerList[i];
		if (layer->role ==OUTPUT){
			if(!doHessVecProd){
				layer->errElem->dyFeatMat = layer->feaElem->yfeatMat;
				calcOutLayerBackwardSignal(layer,anndef);
				}else{
				if(useGNMatrix){
					computeHessOfLossFunc(layer,anndef);
					layer->errElem->dyFeatMat = layer->gnInfo->Ractivations;
				}
			}
			
		}else{
			// from previous iteration dxfeat that is dyfeat now is dE/dZ.. computing dE/da
			computeActivationDrv(layer); 
			computeDrvAct(layer->errElem->dyFeatMat,layer->feaElem->yfeatMat,layer->dim*BATCHSAMPLES); 
		}
		#ifdef CBLAS
		//compute dxfeatMat: the result  should be an array [ b1 b2..] where b1 is one of dim srcDim
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, layer->srcDim, BATCHSAMPLES, layer->dim, 1, layer->weights, layer->srcDim, layer->errElem->dyFeatMat, layer->dim, 0,layer->errElem->dxFeatMat,layer->srcDim);
		//compute derivative with respect to weights: the result  should be an array of array of [ n1 n2] where n1 is of length srcDim
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, layer->srcDim, layer->dim, BATCHSAMPLES, 1, layer->feaElem->xfeatMat, layer->srcDim, layer->errElem->dyFeatMat, layer->dim, 0, layer->traininfo->dwFeatMat, layer->srcDim);
		//compute derivative with respect to bias: the result should an array of size layer->dim ..we just sum the columns of dyFeatMat
		sumColsOfMatrix(layer->errElem->dyFeatMat,layer->traininfo->dbFeaMat,layer->dim,BATCHSAMPLES);
		//rescale dbfeatMat and dWfeatMat
		scaleMatrixOrVec(layer->traininfo->dwFeatMat,(double) 1/BATCHSAMPLES,layer->dim*layer->srcDim);
		scaleMatrixOrVec(layer->traininfo->dbFeaMat,(double) 1/BATCHSAMPLES,layer->dim);
		#endif
	}
}


//----------------------------------------------------------------------------------------------------
/*This section deals with running schedulers to iteratively update the parameters of the neural net**/
//-------------------------------------------------------------------------------------------------------

void fillCache(LELink layer,int dim,Boolean weights){
	#ifdef CBLAS
	if (weights){
		double* paramCache = (double *) getHook(layer->traininfo,1);
		copyMatrixOrVec(layer->weights,paramCache,dim);
	}else{
		double* paramCache = (double *) getHook(layer->traininfo,2);
		copyMatrixOrVec(layer->bias,paramCache,dim);
	}
	#endif
}

void cacheParameters(ADLink anndef){
	int i;
	LELink layer;
	for (i =(anndef->layerNum-1) ; i >=0; --i){
		layer = anndef->layerList[i];
		copyMatrixOrVec(layer->bias,layer->bestBias,layer->dim);
		copyMatrixOrVec(layer->weights,layer->bestweights,layer->dim*layer->srcDim);
		//fillCache(layer,layer->dim*layer->srcDim,TRUE);
		//fillCache(layer,layer->dim,FALSE);
	}	
	printf("successfully cached best parameters \n");
}

Boolean initialiseParameterCaches(ADLink anndef){
	int i;
	LELink layer;

	for (i =(anndef->layerNum-1) ; i >=0; --i){
		layer = anndef->layerList[i];
		layer->bestweights = malloc(sizeof(double)*(layer->dim*layer->srcDim));
		layer->bestBias = malloc(sizeof(double)*layer->dim);
		/*double* weightCache = malloc(sizeof(double)*(layer->dim*layer->srcDim));
		double* biasCache =  malloc(sizeof(double)*layer->dim);
		setHook(layer->traininfo,weightCache,1);
		setHook(layer->traininfo,biasCache,2);*/
	}	
	printf("successfully intialised caches \n");
	return TRUE;
	 
}

void perfBinClassf(double *yfeatMat, double *predictions,int dataSize){
	int i;
	for (i = 0; i< dataSize;i++){
		predictions[i] = yfeatMat[i]>0.5 ? 1 :0;
		printf("Predictions %d  %lf  and yfeat is %lf and real predict is %lf \n",i,predictions[i],yfeatMat[i],validationLabelIdx[i]);
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
/** the function calculates the average error*/
void updatateAcc(double *labels, LELink layer,int dataSize){
	int i, dim;
	double accCount,holdingVal;
	accCount=0;
	if (anndef->target==CLASSIFICATION){
		double *predictions = malloc(sizeof(double)*dataSize);
		if (layer->dim >1){
			dim = layer->dim;
			findMaxElement(layer->feaElem->yfeatMat,dataSize,dim,predictions);
		}else{
			perfBinClassf(layer->feaElem->yfeatMat,predictions,dataSize);
		}
		for (i = 0; i<dataSize;i++){
			if (fabs(predictions[i]-labels[i])>0.01){
				accCount+=1;
			}	
		}
		free(predictions);
	}else{
		subtractMatrix(layer->feaElem->yfeatMat, labels, dataSize);
		for (i = 0;i<dataSize*layer->dim;i++){
			holdingVal = layer->feaElem->yfeatMat[i];
			accCount+= holdingVal*holdingVal;
		}
	}		
		
	modelSetInfo->crtVal = accCount/dataSize;
	printf("The critical value is %f  %d\n", modelSetInfo->crtVal,dataSize);
}
void updateNeuralNetParams(ADLink anndef, double lrnrate, double momentum, double weightdecay){
	int i;
	LELink layer;
	for (i =(anndef->layerNum-1) ; i >=0; --i){
		layer = anndef->layerList[i];
		//if we have a regularised error function: 
		if (weightdecay > 0){
			//printf("SHIOULD NOTT REACH HER \n");
			/** here we are computing delE/w + lambda w and then later we add leanring rate -mu(delE/w + lambda w)**/
			addMatrixOrVec(layer->weights,layer->traininfo->dwFeatMat,layer->dim*layer->srcDim,weightdecay);
			addMatrixOrVec(layer->bias,layer->traininfo->dbFeaMat,layer->dim,weightdecay);
		}
		if (momentum > 0 ){
			scaleMatrixOrVec(layer->traininfo->updatedWeightMat,momentum,layer->dim*layer->srcDim);
			scaleMatrixOrVec(layer->traininfo->updatedBiasMat,momentum,layer->dim);
			addMatrixOrVec(layer->traininfo->dwFeatMat,layer->traininfo->updatedWeightMat,layer->dim*layer->srcDim,lrnrate);
			addMatrixOrVec(layer->traininfo->dbFeaMat,layer->traininfo->updatedBiasMat,layer->dim,1-momentum);
			//updating parameters: first we need to descale the lambda from weights and bias
			addMatrixOrVec(layer->traininfo->updatedWeightMat,layer->weights,layer->dim*layer->srcDim,1);
			addMatrixOrVec(layer->traininfo->updatedBiasMat,layer->bias,layer->dim,1);
		}else{
			//updating parameters: first we need to descale the lambda from weights and bias
			addMatrixOrVec(layer->traininfo->dwFeatMat,layer->weights,layer->dim*layer->srcDim,lrnrate);
			addMatrixOrVec(layer->traininfo->dbFeaMat,layer->bias,layer->dim,lrnrate);
		}
	}
		
}

void updateLearningRate(int currentEpochIdx, double *lrnrate){
	double crtvaldiff;
	if (currentEpochIdx == 0) {
		*lrnrate = (-1)* initLR;
	}else if (modelSetInfo !=NULL){
		crtvaldiff = (modelSetInfo->crtVal - modelSetInfo->prevCrtVal);
		if (crtvaldiff > threshold){
			*lrnrate /=2;
			printf("Learning rate has been halved !! \n");
		}
	}
}

Boolean terminateSchedNotTrue(int currentEpochIdx,double lrnrate){
	printf("lrn rate %f\n",lrnrate);
	if (currentEpochIdx == 0) return TRUE;
	if (currentEpochIdx >=0 && currentEpochIdx >= maxEpochNum){
		printf("Stops because max epoc has been reached \n");
		return FALSE;
	}
	if( (-1*lrnrate) < minLR){
		printf("Stopped :lrnrate <minLR\n");
		return FALSE;
	}
	if (fabs(lrnrate-0)<0.0001) {
		printf("Stopped :lrn rate is too small\n");
		return FALSE;
	}

	return TRUE; 
}

void TrainDNNGD(){
	int currentEpochIdx;
	double learningrate;
	
	currentEpochIdx = 0;
	learningrate = 0;
	modelSetInfo = malloc (sizeof(MSI));
	modelSetInfo->crtVal = 0;
	modelSetInfo->bestValue = DBL_MAX;
	modelSetInfo->prevCrtVal = 0;

	//with the initialisation of weights,check how well DNN performs on validation data
	setBatchSize(validationDataSetSize);
	reinitLayerFeaMatrices(anndef);
	//load  entire batch into neuralNet
	loadDataintoANN(validationData,validationLabelIdx);
	fwdPassOfANN(anndef);
	printf("successfully performed forward pass of DNN on validation data\n");
	updatateAcc(validationLabelIdx, anndef->layerList[numLayers-1],BATCHSAMPLES);
	printf("successfully accumulated counts  \n");
	
	initialiseParameterCaches(anndef);
	if(modelSetInfo->crtVal < modelSetInfo->bestValue){
		cacheParameters(anndef);
		modelSetInfo->bestValue = modelSetInfo->crtVal;
	}
	while(terminateSchedNotTrue(currentEpochIdx,learningrate)){
		printf("epoc number %d \n", currentEpochIdx);
		updateLearningRate(currentEpochIdx,&learningrate);
		//load training data into the ANN and perform forward pass
		setBatchSize(trainingDataSetSize);
		reinitLayerFeaMatrices(anndef);
		loadDataintoANN(inputData,labels);
		fwdPassOfANN(anndef);
		
		// run backpropagation and update the parameters:
		backPropBatch(anndef,FALSE);
		printf("computed gradients for epoch %d\n",currentEpochIdx);
		updateNeuralNetParams(anndef,learningrate,momentum,weightdecay);
		//forward pass of DNN on validation data if VD is provided
		/**check the new performance of the updated neural net against the validation data set*/
		setBatchSize(validationDataSetSize);
		reinitLayerFeaMatrices(anndef);
		//perform forward pass on validation data and check the performance of the DNN on the validation dat set
		loadDataintoANN(validationData,validationLabelIdx);
		fwdPassOfANN(anndef);
		modelSetInfo->prevCrtVal = modelSetInfo->crtVal;
		printf("previous crt value %lf\n ",modelSetInfo->prevCrtVal);
		updatateAcc(validationLabelIdx,anndef->layerList[numLayers-1],validationDataSetSize);
		if (modelSetInfo->crtVal < modelSetInfo->bestValue){
			cacheParameters(anndef);
			modelSetInfo->bestValue = modelSetInfo->crtVal;
		}
		currentEpochIdx+=1;
	}
	printf("The minimum error on the validation data set is %lf percent \n",modelSetInfo->bestValue*100);
}

//----------------------------------------------------------------------------------------------------------
/**this segment of the code is reponsible for accumulating the gradients **/
//---------------------------------------------------------------------------------------------------------
void setHook(Ptr m, Ptr ptr,int incr){
	Ptr *p;
	p = (Ptr *) m; 
	p -= incr; 
  *p = ptr;
 }

Ptr getHook(Ptr m,int incr){
	Ptr *p;
  p = (Ptr *) m; p -=incr; return *p;
}

void accumulateLayerGradient(LELink layer,double weight){
	assert(layer->traininfo->updatedBiasMat != NULL);
	assert(layer->traininfo->updatedWeightMat != NULL);
	#ifdef CBLAS
	cblas_dcopy(layer->srcDim*layer->dim,layer->traininfo->dwFeatMat,1,layer->traininfo->updatedWeightMat,1);
	cblas_dcopy(layer->dim,layer->traininfo->dbFeaMat,1,layer->traininfo->updatedBiasMat,1);
	#else
	//CPU version
	int i,j;
	for (i = 0; i<layer->dim;i++){
		*(layer->traininfo->updatedBiasMat+i) = layer->traininfo->dbFeaMat[i];
		for (j =0; j<layer->srcDim;j++){
			*(layer->traininfo->updatedWeightMat +i*layer->srcDim +j) = layer->traininfo->dwFeatMat[i*layer->srcDim +j];
		}
	}
	#endif
}

void accumulateGradientsofANN(ADLink anndef){
	int i;
	LELink layer;
	for (i = 0; i< anndef->layerNum;i++){
		layer = anndef->layerList[i];
		accumulateLayerGradient(layer,1);
	}
}
//-----------------------------------------------------------------------------------------------------------------------------
/*This section of the code is respoonsible for computing the directional derivative of the error function using forward differentiation*/
//---------------------------------------------------------------------------------------------------------------------------------
/**given a vector in parameteric space, this function copies the the segment of the vector that aligns with the parameters of the given layer*/
void setParameterDirections(double * weights, double* bias, LELink layer){
	assert(layer->gnInfo !=NULL);
	copyMatrixOrVec(weights,layer->gnInfo->vweights,layer->dim*layer->srcDim);
	copyMatrixOrVec(bias,layer->gnInfo->vbiases,layer->dim);
}

void addTikhonovDamping(ADLink anndef){
	int i; 
	LELink layer;
	for (i = 0; i < anndef->layerNum; i++){
		layer = anndef->layerList[i]; 
		addMatrixOrVec(layer->gnInfo->vweights,layer->traininfo->dwFeatMat,layer->dim*layer->srcDim,weightdecay);
		addMatrixOrVec(layer->gnInfo->vbiases,layer->traininfo->dbFeaMat,layer->dim,weightdecay);
	}		
}

void setSearchDirectionCG(ADLink anndef, Boolean Parameter){
	int i; 
	LELink layer;
	for (i = 0; i < anndef->layerNum; i++){
		layer = anndef->layerList[i]; 
		if(Parameter){
			setParameterDirections(layer->cgInfo->searchDirectionUpdateWeights,layer->cgInfo->searchDirectionUpdateBias,layer);
		}else{
			scaleMatrixOrVec(layer->cgInfo->delweightsUpdate,0.95,layer->dim*layer->srcDim);
			scaleMatrixOrVec(layer->cgInfo->delbiasUpdate,0.95,layer->dim);
			setParameterDirections(layer->cgInfo->delweightsUpdate,layer->cgInfo->delbiasUpdate,layer);
		}
	}
}

/**this function computes R(z) = h'(a)R(a)  : we assume h'a is computed during computation of gradient**/
void updateRactivations(LELink layer){
	//CPU Version
	int i;
	for (i = 0; i < layer->dim*BATCHSAMPLES; i++){
		switch (layer->actfuncKind){
			case SIGMOID:
				layer->gnInfo->Ractivations[i] = layer->gnInfo->Ractivations[i]* (layer->feaElem->yfeatMat[i])*(1-layer->feaElem->yfeatMat[i]);
		
			default:
				layer->gnInfo->Ractivations[i]*=1; 
		}
		
	}
}

/** this function compute \sum wji R(zi)-previous layer and adds it to R(zj)**/
void computeRactivations(LELink layer){
	#ifdef CBLAS
		int i,off;
		for (i = 0, off = 0; i < BATCHSAMPLES;i++, off += layer->dim){
			 cblas_daxpy(layer->dim,1,layer->bias,1,layer->gnInfo->Ractivations + off,1);
		}
		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, layer->dim, BATCHSAMPLES, layer->srcDim, 1, layer->weights, layer->srcDim, layer->src->gnInfo->Ractivations, layer->srcDim, 1.0, layer->gnInfo->Ractivations, layer->dim);
	#endif
}

/**this function computes sum vji xi */
void computeVweightsProjection(LELink layer){
	int i,off;
	for (i = 0, off = 0; i < BATCHSAMPLES;i++, off += layer->dim){
			copyMatrixOrVec(layer->gnInfo->vbiases,layer->gnInfo->Ractivations + off ,layer->dim);
	}
	#ifdef CBLAS
		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, layer->dim, BATCHSAMPLES, layer->srcDim, 1, layer->gnInfo->vweights, layer->srcDim, layer->feaElem->xfeatMat, layer->srcDim, 0, layer->gnInfo->Ractivations, layer->dim);
	#endif
}
void computeDirectionalErrDrvOfLayer(LELink layer, int layerid){
	if (layerid == 0){
		/**compute sum vji xi **/
		computeVweightsProjection(layer);
		/** compute R(z) = h'(a)* R(a)**/
		//note h'(a) is already computed during backprop of gradients;
		updateRactivations(layer);
	}else{
		computeVweightsProjection(layer);
		/** compute R(z) = h'(a)* R(a)**/
		computeRactivations(layer);
		if (layer->role != OUTPUT){
			updateRactivations(layer);
		}
	}
}

void computeDirectionalErrDerivativeofANN(ADLink anndef){
	int i;
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		computeDirectionalErrDrvOfLayer(layer,i);
	}
}
//--------------------------------------------------------------------------------------------------------
/**This section of the code implements  The conjugate Gradient algorithm **/
void updateParameterDir(ADLink anndef,double * residueDotProductResult, double *prevresidueDotProductResult){
	int i; 
	LELink layer;
	double beta_w = residueDotProductResult[0]/prevresidueDotProductResult[0];
	double beta_b = residueDotProductResult[1]/prevresidueDotProductResult[1];
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		//first we set p_K+1 = beta  p_k then we set p_k+1+=-r_k+1
		scaleMatrixOrVec(layer->cgInfo->searchDirectionUpdateWeights,beta_w,layer->dim*layer->srcDim);
		addMatrixOrVec(layer->cgInfo->residueUpdateWeights,layer->cgInfo->searchDirectionUpdateWeights,layer->dim*layer->srcDim,-1);
		
		scaleMatrixOrVec(layer->cgInfo->searchDirectionUpdateBias,beta_b,layer->dim);
		addMatrixOrVec(layer->cgInfo->residueUpdateBias,layer->cgInfo->searchDirectionUpdateBias,layer->dim,-1);
	}	
}

void updateResidue(ADLink anndef,double *residueDotProductResult, double *searchDotProductResult){
	int i; 
	LELink layer;
	double alpha_w = residueDotProductResult[0]/searchDotProductResult[0];
	double alpha_b = residueDotProductResult[1]/searchDotProductResult[1];
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		addMatrixOrVec(layer->traininfo->dwFeatMat,layer->cgInfo->residueUpdateWeights,layer->dim*layer->srcDim,alpha_w);
		addMatrixOrVec(layer->traininfo->dbFeaMat,layer->cgInfo->residueUpdateBias,layer->dim,alpha_b);
	}	
}	

void updatedelParameters(double * residueDotProductResult, double *searchDotProductResult){
	int i; 
	LELink layer;
	double alpha_w = residueDotProductResult[0]/searchDotProductResult[0];
	double alpha_b = residueDotProductResult[1]/searchDotProductResult[1];
	for (i = 0; i<anndef->layerNum;i++){
		printf("LAYER ID %d >>>>>>\n",i);
		layer = anndef->layerList[i];
		addMatrixOrVec(layer->cgInfo->searchDirectionUpdateWeights,layer->cgInfo->delweightsUpdate,layer->dim*layer->srcDim,alpha_w);
		addMatrixOrVec(layer->cgInfo->searchDirectionUpdateBias,layer->cgInfo->delbiasUpdate,layer->dim,alpha_b);
		printf("printing  Search Dir W >>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
		printMatrix(layer->cgInfo->searchDirectionUpdateWeights,layer->dim,layer->srcDim);
		printf("printing  Search dir B>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
		printMatrix(layer->cgInfo->searchDirectionUpdateBias,1,layer->dim);


		printf("printing  DEL W >>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
		printMatrix(layer->cgInfo->delweightsUpdate,layer->dim,layer->srcDim);
		printf("printing  DEL B >>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
		printMatrix(layer->cgInfo->delbiasUpdate,1,layer->dim);
		
	}		
			
}

 void computeSearchDirMatrixProduct( ADLink anndef,double * searchVecMatrixVecProductResult){
	int i; 
	double weightsum = 0;
	double biasSum = 0 ;
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		#ifdef CBLAS
		weightsum+= cblas_ddot(layer->dim*layer->srcDim,layer->cgInfo->searchDirectionUpdateWeights,1,layer->traininfo->dwFeatMat,1);
		biasSum += cblas_ddot(layer->dim,layer->cgInfo->searchDirectionUpdateBias,1,layer->traininfo->dbFeaMat,1);
		#endif
	}	
	searchVecMatrixVecProductResult[0] = weightsum;
	searchVecMatrixVecProductResult[1] = biasSum;
	
}

void computeSearchDirDotProduct(ADLink anndef, double *searchDotProductResult){
	int i; 
	double weightsum = 0;
	double biasSum = 0 ;
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		#ifdef CBLAS
		weightsum+= cblas_ddot(layer->dim*layer->srcDim,layer->cgInfo->searchDirectionUpdateWeights,1,layer->cgInfo->searchDirectionUpdateWeights,1);
		biasSum += cblas_ddot(layer->dim,layer->cgInfo->searchDirectionUpdateBias,1,layer->cgInfo->searchDirectionUpdateBias,1);
		#endif
	
	}
	searchDotProductResult[0] = weightsum;
	searchDotProductResult[1] = biasSum;
}

	
 void computeResidueDotProduct(ADLink anndef, double * residueDotProductResult){
	int i; 
	double weightsum = 0;
	double biasSum = 0;
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		#ifdef CBLAS
		weightsum+= cblas_ddot(layer->dim*layer->srcDim,layer->cgInfo->residueUpdateWeights,1,layer->cgInfo->residueUpdateWeights,1);
		biasSum += cblas_ddot(layer->dim,layer->cgInfo->residueUpdateBias,1,layer->cgInfo->residueUpdateBias,1);
		#endif
	
	}
	residueDotProductResult[0] = weightsum;
	residueDotProductResult[1] = biasSum;

}
void normaliseSearchDirections(ADLink anndef){
	int i; 
	LELink layer;
	double dotProduct[] ={ 0,0};
	computeSearchDirDotProduct(anndef,dotProduct);
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		scaleMatrixOrVec(layer->cgInfo->searchDirectionUpdateWeights, 1/sqrt(dotProduct[0]) ,layer->dim*layer->srcDim);
		scaleMatrixOrVec(layer->cgInfo->searchDirectionUpdateBias, 1/sqrt(dotProduct[1]) ,layer->dim);
	}	
}

void reInitialiseResidueaAndSearchDirection(ADLink anndef){
	int i; 
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		copyMatrixOrVec(layer->traininfo->dwFeatMat,layer->cgInfo->residueUpdateWeights,layer->dim*layer->srcDim);
		copyMatrixOrVec(layer->traininfo->dbFeaMat,layer->cgInfo->residueUpdateBias,layer->dim);
			
		addMatrixOrVec(layer->traininfo->updatedWeightMat,layer->cgInfo->residueUpdateWeights,layer->dim*layer->srcDim, 1);
		addMatrixOrVec(layer->traininfo->updatedBiasMat,layer->cgInfo->residueUpdateBias,layer->dim, 1);

		//initialising the search direction
		copyMatrixOrVec(layer->cgInfo->residueUpdateWeights,layer->cgInfo->searchDirectionUpdateWeights,layer->dim*layer->srcDim);
		copyMatrixOrVec(layer->cgInfo->residueUpdateBias,layer->cgInfo->searchDirectionUpdateBias,layer->dim);
			
		scaleMatrixOrVec(layer->cgInfo->searchDirectionUpdateWeights, -1 ,layer->dim*layer->srcDim);
		scaleMatrixOrVec(layer->cgInfo->searchDirectionUpdateBias, -1 ,layer->dim);
	}
}

void initialiseResidueaAndSearchDirection(ADLink anndef){
	int i; 
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];

		//ro = A*0 +b 
		copyMatrixOrVec(layer->traininfo->updatedWeightMat, layer->cgInfo->residueUpdateWeights,layer->dim*layer->srcDim);
		copyMatrixOrVec(layer->traininfo->updatedWeightMat, layer->cgInfo->searchDirectionUpdateWeights,layer->dim*layer->srcDim);
		
		copyMatrixOrVec(layer->traininfo->updatedBiasMat,layer->cgInfo->residueUpdateBias,layer->dim);
		copyMatrixOrVec(layer->traininfo->updatedBiasMat,layer->cgInfo->searchDirectionUpdateBias,layer->dim);
				
		scaleMatrixOrVec(layer->cgInfo->searchDirectionUpdateWeights, -1 ,layer->dim*layer->srcDim);
		scaleMatrixOrVec(layer->cgInfo->searchDirectionUpdateBias, -1 ,layer->dim);
	}
}

void runConjugateGradient(Boolean firstEverRun){
	int numberofRuns = 0;
	double *residueDotProductResult;
	double *prevresidueDotProductResult;
	double *searchVecMatrixVecProductResult;

	residueDotProductResult = malloc(sizeof(double)*2);
	prevresidueDotProductResult = malloc(sizeof(double)*2);
	searchVecMatrixVecProductResult = malloc(sizeof(double)*2);	
	if (firstEverRun){
		printf("First ever run of CG\n");
		initialiseResidueaAndSearchDirection(anndef);
		normaliseSearchDirections(anndef);
	}else{
		printf("using previous CG iterates\n");
		//compute A*del w_0
		setSearchDirectionCG(anndef,FALSE);
		computeDirectionalErrDerivativeofANN(anndef);
		backPropBatch(anndef,TRUE);
		addTikhonovDamping(anndef);
		//setting r_0 = A*del w +  b and p_0 = -r0
		reInitialiseResidueaAndSearchDirection(anndef);
		normaliseSearchDirections(anndef);
	}
	while(numberofRuns < maxNumOfCGruns){
		printf("RUN  %d >>>>>>>>>>>>>>>>>>>>>\n",numberofRuns);
		//----Compute Gauss Newton Matrix product	
		//set v
		setSearchDirectionCG(anndef,TRUE);
		// compute Jv i.e 
		computeDirectionalErrDerivativeofANN(anndef);
		//compute J^T del L^2 J v i.e A p_k
		backPropBatch(anndef,TRUE);
		addTikhonovDamping(anndef);
		//---------------------------
		//compute r_k^T r_k
		computeResidueDotProduct(anndef, residueDotProductResult);
		//compute p_k^T A p_k
		computeSearchDirMatrixProduct(anndef,searchVecMatrixVecProductResult);
		//update del w_k+1  = del w_k + alpha pk
		updatedelParameters(residueDotProductResult,searchVecMatrixVecProductResult);
		//update r_k+1 = r_k + alpha A p_k
		updateResidue(anndef,residueDotProductResult,searchVecMatrixVecProductResult);

		prevresidueDotProductResult[0] = residueDotProductResult[0];
		prevresidueDotProductResult[1] = residueDotProductResult[1];
		//compute r_(k+1)^T r_(k+1)
		computeResidueDotProduct(anndef, residueDotProductResult);
		//compute p_(k+1) = -r_k+1 + beta p_k
		updateParameterDir(anndef,residueDotProductResult,prevresidueDotProductResult);
		normaliseSearchDirections(anndef);
		numberofRuns+=1;
	}

	int i; 
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		printf("LAYER ID %d >>>>>>\n",i);
		layer = anndef->layerList[i];
		printf("printing  DEL W >>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
		printMatrix(layer->cgInfo->delweightsUpdate,layer->dim,layer->srcDim);
		printf("printing  DEL B >>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
		printMatrix(layer->cgInfo->delbiasUpdate,1,layer->dim);
		

	}	

	free(residueDotProductResult);
	free(prevresidueDotProductResult);
	free(searchVecMatrixVecProductResult);
}

//------------------------------------------------------------------------------------------
/*This section of the code performs HF training**/
void updateNeuralNetParamsHF( ADLink anndef, double lrnRate){
	int i;
	LELink layer;
	for (i =(anndef->layerNum-1) ; i >=0; --i){
		layer = anndef->layerList[i];
		addMatrixOrVec(layer->cgInfo->delweightsUpdate,layer->weights,layer->dim*layer->srcDim,lrnRate);
		addMatrixOrVec(layer->cgInfo->delbiasUpdate,layer->bias,layer->dim,lrnRate);
		printf(" WEIGHTS  for layer id %d >>>>>>>>>>>>>\n",i);
		printMatrix(layer->weights,layer->dim,layer->srcDim);	

		printf(" BIAS >>>>>>>>>>>>>\n");
		printMatrix(layer->bias,1,layer->dim);	
	}
}	

void TrainDNNHF(){
	int currentEpochIdx;
	double learningrate;
	currentEpochIdx = 0;
	learningrate = 0;
	modelSetInfo = malloc (sizeof(MSI));
	modelSetInfo->crtVal = 0;
	modelSetInfo->bestValue = DBL_MAX;
	modelSetInfo->prevCrtVal = 0;
	
	//with the initialisation of weights,check how well DNN performs on validation data
	
	setBatchSize(validationDataSetSize);
	reinitLayerFeaMatrices(anndef);
	//load  entire batch into neuralNet
	loadDataintoANN(validationData,validationLabelIdx);
	fwdPassOfANN(anndef);
	printf("successfully performed forward pass of DNN on validation data\n");
	updatateAcc(validationLabelIdx, anndef->layerList[numLayers-1],BATCHSAMPLES);
	printf("successfully accumulated counts  \n");
	
	initialiseParameterCaches(anndef);
	if(modelSetInfo->crtVal<modelSetInfo->bestValue){
		cacheParameters(anndef);
		modelSetInfo->bestValue = modelSetInfo->crtVal;
	}
	while(terminateSchedNotTrue(currentEpochIdx,learningrate)){
		printf("epoc number %d \n", currentEpochIdx);
		updateLearningRate(currentEpochIdx,&learningrate);
		/*for each iteration: compute and  accumulate the gradient and then run CG*/
		setBatchSize(trainingDataSetSize);
		reinitLayerFeaMatrices(anndef);
		loadDataintoANN(inputData,labels);
		fwdPassOfANN(anndef);
		// run backpropagation and compute gradients
		backPropBatch(anndef,FALSE);
		accumulateGradientsofANN(anndef);
		printf("successfully accumulated Gradients \n");
		if (currentEpochIdx == 0){
			runConjugateGradient(TRUE);
		}else{
			runConjugateGradient(FALSE);
		}
		printf("successfully completed a run of CG \n");
		printf("DOES REACH HERE \n");
		updateNeuralNetParamsHF(anndef,learningrate);
		printf("DOESNT REACH HERE \n");
		/**check the new performance of the updated neural net against the validation data set*/
		setBatchSize(validationDataSetSize);

		reinitLayerFeaMatrices(anndef);
		printf("DOES REACH HERE  as well\n");
		//perform forward pass on validation data and check the performance of the DNN on the validation dat set
		loadDataintoANN(validationData,validationLabelIdx);
		fwdPassOfANN(anndef);
		printf("DOES REACH HERE  as well 2 \n");
		printYfeat(anndef,1);
		modelSetInfo->prevCrtVal = modelSetInfo->crtVal;
		updatateAcc(validationLabelIdx,anndef->layerList[numLayers-1],validationDataSetSize);
		if (modelSetInfo->crtVal < modelSetInfo->bestValue){
			cacheParameters(anndef);
			modelSetInfo->bestValue = modelSetInfo->crtVal;
		}
		currentEpochIdx+=1;
	}
	printf("The minimum error on the validation data set is %lf percent \n",modelSetInfo->bestValue*100);

}

//========================================================================================================
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
				if (anndef->layerList[i]->traininfo!=NULL){
					free (anndef->layerList[i]->traininfo->dwFeatMat);
					free (anndef->layerList[i]->traininfo->dbFeaMat);
					if(anndef->layerList[i]->traininfo->updatedBiasMat !=NULL){
						free(anndef->layerList[i]->traininfo->updatedBiasMat);
					}
					if (anndef->layerList[i]->traininfo->updatedWeightMat!= NULL){
						free(anndef->layerList[i]->traininfo->updatedWeightMat);
					}
					free (anndef->layerList[i]->traininfo);
				}
				if (anndef->layerList[i]->weights !=NULL){
					free (anndef->layerList[i]->weights);
				}
				if (anndef->layerList[i]->bias !=NULL){
					free (anndef->layerList[i]->bias);
				}
				if(anndef->layerList[i]->gnInfo != NULL){
					if (anndef->layerList[i]->gnInfo->vweights !=NULL){
						free(anndef->layerList[i]->gnInfo->vweights);
					}
					if (anndef->layerList[i]->gnInfo->vbiases !=NULL){
						free (anndef->layerList[i]->gnInfo->vbiases);
					}
					if (anndef->layerList[i]->gnInfo->Ractivations !=NULL){
						free(anndef->layerList[i]->gnInfo->Ractivations);
					}
					free (anndef->layerList[i]->gnInfo);
				}
				if(anndef->layerList[i]->cgInfo !=NULL){
					if(anndef->layerList[i]->cgInfo->delweightsUpdate != NULL){
						free(anndef->layerList[i]->cgInfo->delweightsUpdate);
					}
					if (anndef->layerList[i]->cgInfo->delbiasUpdate != NULL){
						free (anndef->layerList[i]->cgInfo->delbiasUpdate);
					}
					if (anndef->layerList[i]->cgInfo->residueUpdateWeights != NULL){
						free(anndef->layerList[i]->cgInfo->residueUpdateWeights );
					}
					if (anndef->layerList[i]->cgInfo->residueUpdateBias != NULL){
						free(anndef->layerList[i]->cgInfo->residueUpdateBias);
					}
					if (anndef->layerList[i]->cgInfo->searchDirectionUpdateBias != NULL){
						free(anndef->layerList[i]->cgInfo->searchDirectionUpdateBias);
					}
					if (anndef->layerList[i]->cgInfo->searchDirectionUpdateWeights != NULL){
						free(anndef->layerList[i]->cgInfo->searchDirectionUpdateWeights);
					}
					free(anndef->layerList[i]->cgInfo);
				}
				free (anndef->layerList[i]);
			}
		}
		free(anndef->layerList);	
		free(anndef);
		free(modelSetInfo);
	}
	free(inputData);
	free(labels);
	free(validationData);
	free(validationLabelIdx);
	printf("Finished freeing memory\n");
}



void printWeights(ADLink anndef, int i){
	LELink layer;
	int k,j; 
	layer = anndef->layerList[i];
	int dim;
	if (layer->dim >10 ) {
		dim= 20;
	} else{
		dim =10;
	}
	printf("printing WEIGHTS \n");
		for (j = 0 ; j< dim;j++){
			for (k =0 ; k <20;k++){
				printf(" %lf ", layer->weights[j*layer->dim +k]);
			}
			printf("\n");
		}
		printf("printing BIAS \n");
		for (j = 0 ; j< layer->dim;j++){
			printf(" %lf ", layer->bias[j]);
		}
		printf("\n");

}

void printDBWeights(ADLink anndef, int i){
	LELink layer;
	int k,j; 
	layer = anndef->layerList[i];
	int dim;
	if (layer->dim >10 ) {
		dim= 20;
	} else{
		dim =10;
	}
	printf("printing WEIGHTS \n");
		for (j = 0 ; j< dim;j++){
			for (k =0 ; k <20;k++){
				printf(" %lf ", layer->traininfo->dwFeatMat[j*layer->dim +k]);
			}
			printf("\n");
		}
		printf("printing BIAS \n");
		for (j = 0 ; j< layer->dim;j++){
			printf(" %lf ", layer->traininfo->dbFeaMat[j]);
		}
		printf("\n");

}

void printMatrix(double * matrix,int row,int col){
int k,j;

int r = row >10 ? 10: row;
int c  = col >10 ? 10:col;
for (j = 0 ; j< r;j++){
	for (k =0 ; k <c;k++){
		printf(" %lf ", matrix[j*row +k]);
		}
	printf("\n");
}

}



void printYfeat(ADLink anndef,int id){
	int i,k,j; 
	LELink layer;
	
		layer = anndef->layerList[id];
		int dim;
	if (layer->dim >10 ) {
		dim= 20;
	} else{
		dim =10;
	}
		printf("layer id %d dim is  %d\n ",id,layer->dim);
		for (i = 0; i<20;i++){
			for (j =0 ; j<dim;j++){
				printf("%lf ",layer->feaElem->yfeatMat[i*layer->dim+j]);
		}
		printf("\n");
	}
	
	printf("printing WEIGHTS \n");
		for (j = 0 ; j< dim;j++){
			for (k =0 ; k <20;k++){
				printf(" %lf ", layer->weights[j*layer->dim +k]);
			}
			printf("\n");
		}
		printf("printing BIAS \n");
		for (j = 0 ; j< layer->dim;j++){
			printf(" %lf ", layer->bias[j]);
		}
		printf("\n");
	
	
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
	trainingDataSetSize = 4;
	setBatchSize(4);
	inputDim = 2;
	inputData =input;

	double test[] = { 5,3,6,8,7,1,10,8};
	validationDataSetSize = 4;
	validationData = test ;

	
	TrainDNNGD();


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


//==============================================================================

int main(int argc, char *argv[]){
	
	  


	/**testing gauss newton product**/
	if (argc != 11 && argc != 13 ){
		printf("The program expects a minimum of  5 args and a maximum of 6 args : Eg : -C config \n -S traindatafile \n -L traininglabels \n -v validationdata \n -vl validationdataLabels \n optional argument : -T testData \n ");
	}
	parseCMDargs(argc, argv);
	/*modelSetInfo = malloc (sizeof(MSI));
	modelSetInfo->crtVal = 0;
	modelSetInfo->bestValue = DBL_MAX;
	modelSetInfo->prevCrtVal = 0;
*/
	doHF = TRUE;
	useGNMatrix = TRUE;
	maxNumOfCGruns = 60 ;
	initialise();
	//TrainDNNGD();
	/*setBatchSize(validationDataSetSize);
	reinitLayerFeaMatrices(anndef);
	loadDataintoANN(validationData,validationLabelIdx);
	

	fwdPassOfANN(anndef);
	
	printYfeat(anndef,2);
		
	updatateAcc(validationLabelIdx, anndef->layerList[numLayers-1],BATCHSAMPLES);

  */
	int i,j,k;
	


	/*LELink layer;
	layer = anndef->layerList[numLayers-1];
	double* yfeatMat = layer->feaElem->yfeatMat;
	printf(" layer id  %d \n",i);
	for (j = 0 ; j<20;j++){
			for (k = 0 ; k <layer->dim;k++){
				printf(" %lf ",yfeatMat[j*layer->dim +k ]);

			}
		printf("\n");

	}
	*/
	/**for ( i = 0; i < BATCHSAMPLES;i ++){
	//	printf(" %lf ", labels[i]);
	} 

	loadDataintoANN(inputData,labels);
	printf("\nINOUT DATA  dim %d  batch sample %d \n", inputDim, BATCHSAMPLES);
	for (j = 0 ; j<20;j++){
				for (k = 0 ; k <40;k++){
					printf(" %lf ",inputData[j*inputDim +k ]);

				}
				//printf("\n");
	}
	printf("\n");
 


	fwdPassOfANN(anndef);
	LELink layer ;
	
	printf("printing OUTPUTS OF EACH LAYER");
	  for (i =0 ; i <anndef->layerNum;i++){
		layer = anndef->layerList[i];
		double* yfeatMat = layer->feaElem->yfeatMat;
		printf(" layer id  %d \n",i);
		for (j = 0 ; j<20;j++){
				for (k = 0 ; k <layer->dim;k++){
					printf(" %lf ",yfeatMat[j*layer->dim +k ]);

				}
				printf("\n");

		}
	}


	double sum =0;
	double test[] ={ 0.780157,  -0.340828 , 0.238752 };
	for (i =0 ; i<3;i ++){
		printf("test %d %lf \n", i, exp(test[i]));
		sum+=exp(test[i]);
	}
	for (i =0 ; i<3;i ++){
		printf("test %d %lf \n", i, exp(test[i])/sum);
		
	}
	
	


  printf("Printing weights\n");
	for ( i = 1; i< numLayers ;i++){
		LELink layer = anndef->layerList[i];
		printf("layer id  %d \n",i);	
		
		for (j = 0 ; j< layer->dim;j++){
			for (k =0 ; k <layer->srcDim;k++){
				printf(" %lf ", layer->weights[j*layer->dim +k]);
			}
			printf("\n");
		}
		for (j = 0 ; j< layer->dim;j++){
			printf(" %lf ", layer->bias[j]);
		}
		printf("\n");

	}

*/
  //TrainDNNGD();

	TrainDNNHF();
	freeMemoryfromANN();

	exit(0);
	printf("HEELLO \n");
	/**


/*	
	parseCfg("cfg");
	doHF =TRUE;
	useGNMatrix =TRUE;
	BATCHSAMPLES=2;
	printf("BATCH SAMPLES is %d \n",BATCHSAMPLES);
	initialiseDNN();

	int i,j,c;
	double V1[] = {1,1,2,1};
	double b1[] ={0,0};
	double V2[] ={1,1,1,1,1,1};
	double b2[] ={0};
	double x[] = {0,1,2,0};
	anndef->layerList[0]->feaElem->xfeatMat = x; 

	double weights[] = { 1,2,1,1};
	double weightsT[] ={1,1,0,2,1,3};
	
	double y[] = {1,1,1,1};
	double y2[] ={ 1,2,3};
	double d[] ={2,1,2,1,4,3,2,3,1};
	double result[] ={0,0,0};
	printf("\n");

	double dyfeat[] ={0.2,10};
	double labels[] ={1.2,11.2};

	

	
	cblas_dgemv(CblasColMajor,CblasNoTrans,3,3,1,d,3,y2,1,0,result,1);
	printf("symmetric matrix vector\n");
	for (i=0;i<3;i++){
		//y[i] = computeSigmoid(y[i]);
		printf(" %lf ", result[i]);
	}
	printf("\n");
		
	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 2, BATCHSAMPLES, 2, 1, weights, 2, x, 2, 0, y, 2);
	
	printf("Test multiplicaiton \n");
	for (i=0;i<4;i++){
		//y[i] = computeSigmoid(y[i]);
		printf(" %lf ", y[i]);
	}
	printf("\n");


	for ( i = 0; i< numLayers ;i++){
		LELink layer = anndef->layerList[i];
		if (i==0){
			setParameterDirections(V1,b1, layer);
		  cblas_dcopy(layer->dim*layer->srcDim, weights, 1,layer->weights, 1);
			
		}	
		else{
			setParameterDirections( V2,b2, layer);
			cblas_dcopy(layer->dim*layer->srcDim, weightsT, 1,layer->weights, 1);
			}  
		printf("layer id  %d \n",i);
		
		
		for (j = 0 ; j< layer->dim;j++){
			for (c =0 ; c <layer->srcDim;c++){
				printf(" %lf ", layer->weights[j*layer->dim +c]);
			}
			layer->bias[j]=0;
			printf("\n");
		}
	}
	printf("doing fwdPass \n");
	fwdPassOfANN(anndef);
	printf(" fwdPass complete \n");
	
	for ( i = 0; i< numLayers ;i++){
		LELink layer = anndef->layerList[i];
		printf("dim is %d \n",layer->dim * BATCHSAMPLES);
		for (j = 0; j<layer->dim*BATCHSAMPLES;j++){
			printf(" %lf ", layer->feaElem->yfeatMat[j]);
		}
		printf("\n");

	}	

	
	printf("doing Jv\n");
	computeDirectionalErrDerivativeofANN(anndef);
	for ( i = 0; i< numLayers ;i++){
		LELink layer = anndef->layerList[i];

		printf("dim is %d \n",layer->dim * BATCHSAMPLES);
		for (j = 0; j<layer->dim*BATCHSAMPLES;j++){
			printf(" %lf ", layer->gnInfo->Ractivations[j]);
		}
		if ( i== (numLayers-1)) {
			computeHessOfLossFunc(layer,anndef);
			printf("The value of del L^2.JV\n");
			for (j = 0; j<layer->dim*BATCHSAMPLES;j++){
				printf(" %lf ", layer->gnInfo->Ractivations[j]);
			}
		}

		printf("\n");
		

	}
	backPropBatch(anndef,TRUE);
	printf("computed Gauss Newton \n");
	for ( i = 0; i< numLayers ;i++){
		LELink layer = anndef->layerList[i];
		printf("layer id  %d \n",i);	
		for (j = 0 ; j< layer->dim;j++){
			for (c =0 ; c <layer->srcDim;c++){
				printf(" %lf ", layer->traininfo->dwFeatMat[j*layer->dim +c]);
			}
			printf("\n");
		}
	}
	weightdecay = 1;
	addTikhonovDamping(anndef);
	printf("The directions\n");
	for ( i = 0; i< numLayers ;i++){
		LELink layer = anndef->layerList[i];
		printf("layer id  %d \n",i);	
		for (j = 0 ; j< layer->dim;j++){
			for (c =0 ; c <layer->srcDim;c++){
				printf(" %lf ", layer->gnInfo->vweights[j*layer->dim +c]);
			}
			printf("\n");
		}
	}	

	printf("After adding Tikhonv damping \n");
	for ( i = 0; i< numLayers ;i++){
		LELink layer = anndef->layerList[i];
		printf("layer id  %d \n",i);	
		for (j = 0 ; j< layer->dim;j++){
			for (c =0 ; c <layer->srcDim;c++){
				printf(" %lf ", layer->traininfo->dwFeatMat[j*layer->dim +c]);
			}
			printf("\n");
		}
	}	
	
	
	freeMemoryfromANN();
	exit(0);

	

	if (argc != 11 && argc != 13 ){
		printf("The program expects a minimum of  5 args and a maximum of 6 args : Eg : -C config \n -S traindatafile \n -L traininglabels \n -v validationdata \n -vl validationdataLabels \n optional argument : -T testData \n ");
	}
	parseCMDargs(argc, argv);
	printf("'REACHES HERE'\n" );


	
	double * ptr  = malloc( sizeof(double));
	double A[] ={1 ,2, 3};
	memcpy(ptr, A, sizeof(double)*3);
	for (i =0 ;i< 3; i++){
		printf(" %lf \n ", ptr[i]);
	}
	free (ptr);




	/**
	c = 0;
	for (i = 0; i < trainingDataSetSize;i++){
		printf("sample  id %d\n",i);
		for (j = 0; j < inputDim; j++){
			printf(" %lf ",inputData[c]);
			c+=1;
		}
		printf("\n");
	}


	c = 0;
	for (i = 0; i < validationDataSetSize;i++){
		printf("sample  id %d\n",i);
		for (j = 0; j < inputDim; j++){
			printf(" %lf ",validationData[c]);
			c+=1;
		}
		printf("\n");
	}
	
	

	//display loaded labels:
	
	c=0;
	for (i = 0; i < trainingDataSetSize;i++){
		printf("sample  id %d\n",i);
		printf("labels \n");
		for (j = 0; j < targetDim; j++){
			printf(" %lf ",labels[c]);
			c+=1;
		}
		printf("\n");
	}
 **/
	/**c=0;
	for (i = 0; i < validationDataSetSize;i++){
		printf("sample  id %d\n",i);
		printf(" validation labels \n");
		for (j = 0; j < targetDim; j++){
			printf(" %lf ",validationLabelIdx[c]);
			c+=1;
		}
		printf("\n");
	}
	**/
 

	//dislay number of units in each hidden layer
	/**
	for ( i = 0 ;  i < numLayers;i++){
		printf("number of hidden units in layer %d  = %d  \n",i,hidUnitsPerLayer[i]);
	}

	//display the activationFunction for each layer:
	for ( i = 0 ;  i < numLayers;i++){
		if (actfunLists[i] == SIGMOID){
			printf( "activation function for layer %i is SIGMOID \n",i);
		}
		if (actfunLists[i] == IDENTITY){
			printf( "activation function for layer %i is IDENTITY \n",i);
		}
		if (actfunLists[i] == TANH){
			printf( "activation function for layer %i is TANH \n",i);
		}
		if (actfunLists[i] == SOFTMAX){
			printf( "activation function for layer %i is SOFTMAX \n",i);
		}
	 }		



	

	printf("maxEpochNum %d\n",maxEpochNum);
	printf("initLR %lf \n",initLR );
	printf("threshold %lf \n",threshold );
	printf("minLR %lf\n",minLR);
	printf("weightdecay %lf\n",weightdecay);
	printf("momentum %lf\n",momentum);

	printf("numLayers %d \n",numLayers);
	printf("trainingDataSize %d\n", trainingDataSetSize);
	printf("validationDataSize %d \n",validationDataSetSize);
	printf("inputDim %d\n",inputDim);
	printf("targetDim %d\n",targetDim );
	printf("numLayers %d\n",numLayers);

	initialise();
	for ( i = 0; i< numLayers ;i++){
		LELink layer = anndef->layerList[i];
		printf("layer id  %d \n",i);	
		for (j = 0 ; j< layer->dim;j++){
			for (c =0 ; c <layer->srcDim;c++){
				printf(" %lf ", layer->weights[j*layer->dim +c]);
			}
			printf("\n");
		}
		for (j = 0 ; j< layer->dim;j++){
			printf(" %lf ", layer->bias[j]);
		}
		printf("\n");

	}


	//fwdPassOfANN(anndef);
	TrainDNNGD();


	for ( i = 0; i< numLayers ;i++){
		LELink layer = anndef->layerList[i];
		printf("layer id  %d \n",i);	
		for (j = 0 ; j< layer->dim;j++){
			for (c =0 ; c <layer->srcDim;c++){
				printf(" %lf ", layer->weights[j*layer->dim +c]);
			}
			printf("\n");
		}
		for (j = 0 ; j< layer->dim;j++){
			printf(" %lf ", layer->bias[j]);
		}
		printf("\n");

	}



	freeMemoryfromANN();
	//unitTests();
	**/
}  

