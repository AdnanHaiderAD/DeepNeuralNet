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
static double * labelMat;
static double * labels;
static double * validationData;
static double * validationLabelIdx;
static int trainingDataSetSize;
static int validationDataSetSize;


/*configurations for DNN architecture*/
static Boolean STOP = FALSE;
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
void loadLabels(double *labelMat, double *labels,char*filepath,char *datatype){
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
			labels[i] = id;
			labelMat[i*targetDim+id] = 1;
			if (i> trainingDataSetSize){
				printf("Error! : the number of training labels doesnt match the size of the training set \n");
				exit(0);
			}
		}else if(strcmp(datatype,"validation")==0){
			id  = strtod(line,NULL);
			labels[i] = id;
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
			printf("parsing training-labels file with trainingDataSize %d \n", trainingDataSetSize);
			labelMat = malloc(sizeof(double)*(trainingDataSetSize*targetDim));
			initialiseWithZero(labelMat,trainingDataSetSize*targetDim);
			labels = malloc(sizeof(double)*(trainingDataSetSize));
			initialiseWithZero(labels,targetDim);
			loadLabels(labelMat,labels,argv[i],"train");
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
			loadLabels(NULL,validationLabelIdx,argv[i],"validation");
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
void loadDataintoANN(double *samples, double *labelMat){
	anndef->layerList[0]->feaElem->xfeatMat = samples;
	anndef->labelMat = labelMat;
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


void subtractMatrix(double *dyfeat, double* labelMat, int dim, double lambda){
	//blas routine
	#ifdef CBLAS
		cblas_daxpy(dim,-lambda,labelMat,1,dyfeat,1);
	#else
	//CPU version
		int i;
		for (i = 0; i<dim;i++){
			dyfeat[i] = dyfeat[i]-lambda*labelMat[i];
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
	double maximum;

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
						maximum = 0;
						for (j = 0; j<layer->dim;j++){
							double value = layer->feaElem->yfeatMat[i*layer->dim+j];
							if (value>maximum) maximum = value;
						}
						for (j = 0; j<layer->dim;j++){
							double value = layer->feaElem->yfeatMat[i*layer->dim+j];
							layer->feaElem->yfeatMat[i*layer->dim+j] = exp(value-maximum);
							sum+= exp(value-maximum);
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
		printf("forward pass of layer in Linear activation%d \n",layer->id);
		printMatrix(layer->feaElem->yfeatMat,BATCHSAMPLES,layer->dim);
		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, layer->dim, BATCHSAMPLES, layer->srcDim, 1, layer->weights, layer->srcDim, layer->feaElem->xfeatMat, layer->srcDim, 1, layer->feaElem->yfeatMat, layer->dim);
		printf("forward pass of layer in linear activation %d \n",layer->id);
		printMatrix(layer->feaElem->yfeatMat,BATCHSAMPLES,layer->dim);
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
					//printf("activation of layer %d before >>>>>",i);
					//printMatrix(layer->feaElem->yfeatMat,BATCHSAMPLES,layer->dim);
					//printf("activation of layer %d after >>>>>",i);
					
					computeNonLinearActOfLayer(layer);
					//printMatrix(layer->feaElem->yfeatMat,BATCHSAMPLES,layer->dim);
	
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
				layer->feaElem->yfeatMat[i] = 1- layer->feaElem->yfeatMat[i]*layer->feaElem->yfeatMat[i];
				//4*layer->feaElem->yfeatMat[i]*(1-layer->feaElem->yfeatMat[i]);
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
		/**extract error directional derivative for a single sample*/ 
		copyMatrixOrVec(layer->gnInfo->Ractivations+i*(layer->dim),RactivationVec,layer->dim);
		copyMatrixOrVec(layer->feaElem->yfeatMat+i*(layer->dim),yfeatVec,layer->dim);
		#ifdef CBLAS
		//compute dia(yfeaVec - yfeacVec*yfeaVec)'
		cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,layer->dim,layer->dim,1,-1,yfeatVec,layer->dim,yfeatVec,1,0,diaP,layer->dim);
		for (j = 0; j<layer->dim;j++){
			diaP[j*(layer->dim+1)] += yfeatVec[j];
		}
		//multiple hessian of loss function of particular sample with Jacobian 
		cblas_dgemv(CblasColMajor,CblasNoTrans,layer->dim,layer->dim,1,diaP,layer->dim,RactivationVec,1,0,result,1);
		#endif
		copyMatrixOrVec(result,layer->feaElem->yfeatMat+i*(layer->dim),layer->dim);
		
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
					subtractMatrix(layer->errElem->dyFeatMat,anndef->labelMat,layer->dim*BATCHSAMPLES,1);
					break;
				case SOFTMAX:
					subtractMatrix(layer->feaElem->yfeatMat,anndef->labelMat,layer->dim*BATCHSAMPLES,1);
					break;
				case TANH:
					break;
				default:
					break;
			}
		break;	
		case (SSE):
			subtractMatrix(layer->errElem->dyFeatMat,anndef->labelMat,layer->dim*BATCHSAMPLES,1);	
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
				calcOutLayerBackwardSignal(layer,anndef);
			}else{
				if(useGNMatrix){
					computeHessOfLossFunc(layer,anndef);
				}
			}
			layer->errElem->dyFeatMat = layer->feaElem->yfeatMat;
			/**normalisation because the cost function is mean log-likelihood*/
			scaleMatrixOrVec(layer->feaElem->yfeatMat,(double)1/BATCHSAMPLES,BATCHSAMPLES*layer->dim);
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
		//scaleMatrixOrVec(layer->traininfo->dwFeatMat,(double) 1/BATCHSAMPLES,layer->dim*layer->srcDim);
		//scaleMatrixOrVec(layer->traininfo->dbFeaMat,(double) 1/BATCHSAMPLES,layer->dim);
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

/**compute the negative mean log likelihood of the training data**/
double computeLogLikelihood(double* output, int batchsamples, int dim , double* labels){
	int i,index;
	double lglikelihood = 0;

	for (i = 0; i < batchsamples ;i++){
		index =  labels[i];
		lglikelihood += log(output[i*dim+index]);
	}

	return -1*(lglikelihood/batchsamples);

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
double updatateAcc(double *labels, LELink layer,int dataSize){
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
		subtractMatrix(layer->feaElem->yfeatMat, labels, dataSize,1);
		for (i = 0;i<dataSize*layer->dim;i++){
			holdingVal = layer->feaElem->yfeatMat[i];
			accCount+= holdingVal*holdingVal;
		}
	}		
		
	return  accCount/dataSize;
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
			addMatrixOrVec(layer->traininfo->dwFeatMat,layer->weights,layer->dim*layer->srcDim,-1*lrnrate);
			addMatrixOrVec(layer->traininfo->dbFeaMat,layer->bias,layer->dim,-1*lrnrate);
		}
	}
		
}

void updateLearningRate(int currentEpochIdx, double *lrnrate){
	double crtvaldiff;
	if (currentEpochIdx == 0) {
		*lrnrate = initLR;
	}else if (modelSetInfo !=NULL){
		crtvaldiff = (modelSetInfo->crtVal - modelSetInfo->prevCrtVal);

		if (crtvaldiff > 0){
			*lrnrate /=2;
			printf("Learning rate has been halved !! \n");
		}
	}
}

Boolean terminateSchedNotTrue(int currentEpochIdx,double lrnrate){
	if (currentEpochIdx == 0) return TRUE;
	if (currentEpochIdx >=0 && currentEpochIdx >= maxEpochNum){
		printf("Stoppped: max epoc has been reached \n");
		return FALSE;
	}
	if( (lrnrate) < minLR){
		printf("Stopped :lrnrate < minLR\n");
		return FALSE;
	}
	return TRUE; 
}

void TrainDNNGD(){
	int currentEpochIdx;
	double learningrate;
	double min_validation_error ;
	
	currentEpochIdx = 0;
	learningrate = 0;
	min_validation_error = 1;

	modelSetInfo = malloc (sizeof(MSI));
	modelSetInfo->crtVal = 0;
	modelSetInfo->bestValue = DBL_MAX;
	modelSetInfo->prevCrtVal = 0;

	//array to store the  training errors and validation error
	double * log_likelihood  = malloc(sizeof(double)*maxEpochNum);
	double * zero_one_error =  malloc(sizeof(double)*maxEpochNum);
	//compute negative-loglikelihood of training data
	printf("computing the mean negative log-likelihood of training data \n");
	loadDataintoANN(inputData,labelMat);
	fwdPassOfANN(anndef);
	log_likelihood[currentEpochIdx] = computeLogLikelihood(anndef->layerList[numLayers-1]->feaElem->yfeatMat,BATCHSAMPLES,anndef->layerList[numLayers-1]->dim,labels);
	modelSetInfo->crtVal = log_likelihood[currentEpochIdx];
	if (modelSetInfo->crtVal < modelSetInfo->bestValue){
		modelSetInfo->bestValue = modelSetInfo->crtVal;
	}
	
	//with the initialisation of weights,checking how well DNN performs on validation data
	printf("computing error on validation data\n");
	setBatchSize(validationDataSetSize);
	reinitLayerFeaMatrices(anndef);
	//load  entire batch into neuralNet
	loadDataintoANN(validationData,validationLabelIdx);
	fwdPassOfANN(anndef);
	zero_one_error[currentEpochIdx] = updatateAcc(validationLabelIdx, anndef->layerList[numLayers-1],BATCHSAMPLES);
	min_validation_error = zero_one_error[currentEpochIdx];
	printf("validation error %lf \n ",zero_one_error[currentEpochIdx] );
		
	initialiseParameterCaches(anndef);
	while(terminateSchedNotTrue(currentEpochIdx,learningrate)){
		printf("epoc number %d \n", currentEpochIdx);
		updateLearningRate(currentEpochIdx,&learningrate);
		//load training data into the ANN and perform forward pass
		setBatchSize(trainingDataSetSize);
		reinitLayerFeaMatrices(anndef);
		loadDataintoANN(inputData,labelMat);
		printf("computing gradient at epoch %d\n", currentEpochIdx);
		fwdPassOfANN(anndef);
		backPropBatch(anndef,FALSE);
		updateNeuralNetParams(anndef,learningrate,momentum,weightdecay);
		fwdPassOfANN(anndef);

		log_likelihood[currentEpochIdx] = computeLogLikelihood(anndef->layerList[numLayers-1]->feaElem->yfeatMat,BATCHSAMPLES,anndef->layerList[numLayers-1]->dim,labels);	
		modelSetInfo->prevCrtVal = modelSetInfo->crtVal;
		modelSetInfo->crtVal = log_likelihood[currentEpochIdx];
		printf("log_likelihood %lf \n ",modelSetInfo->crtVal );
		
		printf("checking the performance of the neural net on the validation data\n");
		setBatchSize(validationDataSetSize);
		reinitLayerFeaMatrices(anndef);
		//perform forward pass on validation data and check the performance of the DNN on the validation dat set
		loadDataintoANN(validationData,validationLabelIdx);
		fwdPassOfANN(anndef);
		zero_one_error[currentEpochIdx] = updatateAcc(validationLabelIdx, anndef->layerList[numLayers-1],BATCHSAMPLES);
		if (modelSetInfo->crtVal < modelSetInfo->bestValue){
			modelSetInfo->bestValue = modelSetInfo->crtVal;
			if (zero_one_error[currentEpochIdx]<min_validation_error){
				min_validation_error = zero_one_error[currentEpochIdx];
				cacheParameters(anndef);
			}
			
		}	
		currentEpochIdx+=1;
		

	}
	printf("TRAINING ERROR >>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
	printVector(log_likelihood,maxEpochNum);
	printf("THE  VALIDATION RESULTS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
	printVector(zero_one_error,maxEpochNum);
	printf("The minimum error on the validation data set is %lf percent  and min log_likelihood is %lf \n",min_validation_error*100, modelSetInfo->bestValue);
	free(zero_one_error);
	free(log_likelihood);

}

//----------------------------------------------------------------------------------------------------------------------------------
/**this segment of the code is reponsible for accumulating the gradients **/
//----------------------------------------------------------------------------------------------------------------------------------

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
	copyMatrixOrVec(layer->traininfo->dwFeatMat, layer->traininfo->updatedWeightMat,layer->srcDim*layer->dim);
	copyMatrixOrVec(layer->traininfo->dbFeaMat, layer->traininfo->updatedBiasMat,layer->dim);
	scaleMatrixOrVec(layer->traininfo->updatedWeightMat, -1 ,layer->dim*layer->srcDim);
	scaleMatrixOrVec(layer->traininfo->updatedBiasMat, -1 ,layer->dim);
		
	
}

void accumulateGradientsofANN(ADLink anndef){
	int i;
	LELink layer;
	for (i = 0; i< anndef->layerNum;i++){
		layer = anndef->layerList[i];
		accumulateLayerGradient(layer,1);
	}
}

//----------------------------------------------------------------------------------------------------------------------------------
//additional functions to check CG sub-routines just in case 
//----------------------------------------------------------------------------------------------------------------------------------

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
void normaliseResidueDirections(ADLink anndef, double* magnitudeOfGradient){
	int i; 
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		scaleMatrixOrVec(layer->cgInfo->residueUpdateWeights, 1/sqrt(magnitudeOfGradient[0]) ,layer->dim*layer->srcDim);
		scaleMatrixOrVec(layer->cgInfo->residueUpdateBias, 1/sqrt(magnitudeOfGradient[1]) ,layer->dim);
	}	

}
void computeNormOfGradient(ADLink anndef){
	int i; 
	double weightsum = 0;
	double biasSum = 0 ;
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		#ifdef CBLAS
		weightsum+= cblas_ddot(layer->dim*layer->srcDim,layer->traininfo->dwFeatMat,1,layer->traininfo->dwFeatMat,1);
		biasSum += cblas_ddot(layer->dim,layer->traininfo->dbFeaMat,1,layer->traininfo->dbFeaMat,1);
		#endif
	
	}
	printf( " The norm  of gradient is %lf %lf  %lf ",weightsum, biasSum,weightsum+biasSum);
}

void computeNormOfAccuGradient(ADLink anndef){
	int i; 
	double weightsum = 0;
	double biasSum = 0 ;
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		#ifdef CBLAS
		weightsum+= cblas_ddot(layer->dim*layer->srcDim,layer->traininfo->updatedWeightMat,1,layer->traininfo->updatedWeightMat,1);
		biasSum += cblas_ddot(layer->dim,layer->traininfo->updatedBiasMat,1,layer->traininfo->updatedBiasMat,1);
		#endif
	
	}
	printf( " The norm  of gradient is %lf %lf  %lf ",weightsum, biasSum,weightsum+biasSum);
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
void normofGV(ADLink anndef){
	int i;
	double weightsum,biasSum =0;
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		#ifdef CBLAS
		weightsum+= cblas_ddot(layer->dim*layer->srcDim,layer->traininfo->dwFeatMat,1,layer->traininfo->dwFeatMat,1);
		biasSum += cblas_ddot(layer->dim,layer->traininfo->dbFeaMat,1,layer->traininfo->dbFeaMat,1);
		#endif
	
	}
	if ((fabs(weightsum-0)<0.000001) || (fabs(biasSum-0)<0.0001)) {STOP =TRUE;}
	printf("the norm of GV is %lf %lf \n", weightsum, biasSum);
}
void normofDELW(ADLink anndef){
	int i;
	double weightsum,biasSum =0;
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		#ifdef CBLAS
		weightsum+= cblas_ddot(layer->dim*layer->srcDim,layer->cgInfo->delweightsUpdate,1,layer->cgInfo->delweightsUpdate,1);
		biasSum += cblas_ddot(layer->dim,layer->cgInfo->delbiasUpdate,1,layer->cgInfo->delbiasUpdate,1);
		#endif
	
	}
	printf("the norm of del W is %lf %lf \n", weightsum, biasSum);
}
//----------------------------------------------------------------------------------------------------------------------------------
/* This section of the code implements HF training*/
//----------------------------------------------------------------------------------------------------------------------------------
void updateNeuralNetParamsHF( ADLink anndef){
	int i;
	LELink layer;
	for (i =(anndef->layerNum-1) ; i >=0; --i){
		layer = anndef->layerList[i];
		addMatrixOrVec(layer->cgInfo->delweightsUpdate,layer->weights,layer->dim*layer->srcDim,1);
		addMatrixOrVec(layer->cgInfo->delbiasUpdate,layer->bias,layer->dim,1);
	}
}
//-----------------------------------------------------------------------------------
/**This section of the code implements the small sub routinesof the  conjugate Gradient algorithm **/


void updateParameterDirection(ADLink anndef,double beta){
	int i; 
	LELink layer;
	
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		//first we set p_K+1 = beta  p_k then we set p_k+1+=r_k+1
		scaleMatrixOrVec(layer->cgInfo->searchDirectionUpdateWeights,beta,layer->dim*layer->srcDim);
		addMatrixOrVec(layer->cgInfo->residueUpdateWeights,layer->cgInfo->searchDirectionUpdateWeights,layer->dim*layer->srcDim,1);
		
		scaleMatrixOrVec(layer->cgInfo->searchDirectionUpdateBias,beta,layer->dim);
		addMatrixOrVec(layer->cgInfo->residueUpdateBias,layer->cgInfo->searchDirectionUpdateBias,layer->dim,1);
	}	
}

void updateResidue(ADLink anndef){
	int i; 
	LELink layer;
	//set del w
	setSearchDirectionCG(anndef,FALSE);
	// compute Jv i.e 
	computeDirectionalErrDerivativeofANN(anndef);
	//compute J^T del L^2 J v i.e A del_w_k+1
	backPropBatch(anndef,TRUE);
	//addTikhonovDamping(anndef);
	//residue r_k+1 = b - A del w_k+1
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		copyMatrixOrVec(layer->traininfo->dwFeatMat,layer->cgInfo->residueUpdateWeights,layer->dim*layer->srcDim);
		copyMatrixOrVec(layer->traininfo->dbFeaMat,layer->cgInfo->residueUpdateBias,layer->dim);
		
		scaleMatrixOrVec(layer->cgInfo->residueUpdateWeights, -1 ,layer->dim*layer->srcDim);
		scaleMatrixOrVec(layer->cgInfo->residueUpdateBias, -1 ,layer->dim);
		
		addMatrixOrVec(layer->traininfo->updatedWeightMat,layer->cgInfo->residueUpdateWeights,layer->dim*layer->srcDim,1);
		addMatrixOrVec(layer->traininfo->updatedBiasMat,layer->cgInfo->residueUpdateBias,layer->dim,1);
	}	
}	
void updatedelParameters(double alpha){
	int i; 
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		addMatrixOrVec(layer->cgInfo->searchDirectionUpdateWeights,layer->cgInfo->delweightsUpdate,layer->dim*layer->srcDim,alpha);
		addMatrixOrVec(layer->cgInfo->searchDirectionUpdateBias,layer->cgInfo->delbiasUpdate,layer->dim,alpha);
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
 	*searchVecMatrixVecProductResult= weightsum + biasSum;
	
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
	*residueDotProductResult = weightsum + biasSum;
	
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
//-----------------------------------------------------------------------------------
/* the following routines compute the directional derivative using forward differentiation*/

/**this function computes R(z) = h'(a)R(a)  : we assume h'a is computed during computation of gradient**/
void updateRactivations(LELink layer){
	//CPU Version
	int i;
	for (i = 0; i < layer->dim*BATCHSAMPLES; i++){
		switch (layer->actfuncKind){
			case SIGMOID:
				layer->gnInfo->Ractivations[i] = layer->gnInfo->Ractivations[i]* (layer->feaElem->yfeatMat[i])*(1-layer->feaElem->yfeatMat[i]);
				break;
			case TANH:
				layer->gnInfo->Ractivations[i] = layer->gnInfo->Ractivations[i]* (1 -layer->feaElem->yfeatMat[i]*layer->feaElem->yfeatMat[i]) ; 
				break;
			default :
				layer->gnInfo->Ractivations[i] = layer->gnInfo->Ractivations[i]* (layer->feaElem->yfeatMat[i])*(1-layer->feaElem->yfeatMat[i]);
				break;
		}
		
	}
}

/** this function compute \sum wji R(zi)-previous layer and adds it to R(zj)**/
void computeRactivations(LELink layer){
	int i,off;
	double * buffer  = malloc (sizeof(double)* BATCHSAMPLES*layer->dim);
	for (i = 0, off = 0; i < BATCHSAMPLES;i++, off += layer->dim){
		copyMatrixOrVec(layer->bias,buffer,layer->dim);
	}
	#ifdef CBLAS
		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, layer->dim, BATCHSAMPLES, layer->srcDim, 1, layer->weights, layer->srcDim, layer->src->gnInfo->Ractivations, layer->srcDim, 1.0, buffer, layer->dim);
	#endif
	addMatrixOrVec(buffer, layer->gnInfo->Ractivations,BATCHSAMPLES*layer->dim, 1);
	free(buffer);
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
		//printf("printing R activations before layer id %d \n",layer->id);
		//printMatrix(layer->gnInfo->Ractivations,BATCHSAMPLES,layer->dim);

		/** compute R(z) = h'(a)* R(a)**/
		//note h'(a) is already computed during backprop of gradients;
		updateRactivations(layer);
		//printf("printing R activations after layer id %d \n",layer->id);
		//printMatrix(layer->gnInfo->Ractivations,BATCHSAMPLES,layer->dim);
	}else{
		/**R(zk) = sum vkz zj**/;
		computeVweightsProjection(layer);
		//printf("printing R activations before layer id %d \n",layer->id);
		//printMatrix(layer->gnInfo->Ractivations,BATCHSAMPLES,layer->dim);

		/** R(zk) += sum wkj Rj **/
		computeRactivations(layer);
		//printf("printing R activations after layer id %d \n",layer->id);
		//printMatrix(layer->gnInfo->Ractivations,BATCHSAMPLES,layer->dim);
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

/**given a vector in parameteric space, this function copies the the segment of the vector that aligns with the parameters of the given layer*/
void setParameterDirections(double * weights, double* bias, LELink layer){
	assert(layer->gnInfo !=NULL);
	copyMatrixOrVec(weights,layer->gnInfo->vweights,layer->dim*layer->srcDim);
	copyMatrixOrVec(bias,layer->gnInfo->vbiases,layer->dim);
}

void setSearchDirectionCG(ADLink anndef, Boolean Parameter){
	int i; 
	LELink layer;
	for (i = 0; i < anndef->layerNum; i++){
		layer = anndef->layerList[i]; 
		if(Parameter){
			setParameterDirections(layer->cgInfo->searchDirectionUpdateWeights,layer->cgInfo->searchDirectionUpdateBias,layer);
		}else{
			setParameterDirections(layer->cgInfo->delweightsUpdate,layer->cgInfo->delbiasUpdate,layer);
		}
	}
}


//-----------------------------------------------------------------------------------

void initialiseResidueaAndSearchDirection(ADLink anndef){
	int i; 
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];

		copyMatrixOrVec(layer->traininfo->updatedWeightMat, layer->cgInfo->residueUpdateWeights,layer->dim*layer->srcDim);
		copyMatrixOrVec(layer->traininfo->updatedWeightMat, layer->cgInfo->searchDirectionUpdateWeights,layer->dim*layer->srcDim);
		
		copyMatrixOrVec(layer->traininfo->updatedBiasMat,layer->cgInfo->residueUpdateBias,layer->dim);
		copyMatrixOrVec(layer->traininfo->updatedBiasMat,layer->cgInfo->searchDirectionUpdateBias,layer->dim);
	}
}

void runConjugateGradient(Boolean firstEverRun){
	int numberofRuns = 0;
	double residueDotProductResult = 0;
	double prevresidueDotProductResult = 0;
	double searchVecMatrixVecProductResult = 0;
	double alpha = 0;
	double beta = 0;
	
	initialiseResidueaAndSearchDirection(anndef);
	while(numberofRuns < maxNumOfCGruns){
		printf("RUN  %d >>>>>>>>>>>>>>>>>>>>>\n",numberofRuns);
		//----Compute Gauss Newton Matrix product	
		//set v
		if(numberofRuns==0){
			setSearchDirectionCG(anndef,TRUE);
		}else{
			setSearchDirectionCG(anndef,FALSE);
		}
		
		//setSearchTemDirectionCG(anndef);
		// compute Jv i.e 
		computeDirectionalErrDerivativeofANN(anndef);
		//compute J^T del L^2 J v i.e A p_k
		backPropBatch(anndef,TRUE);
		//addTikhonovDamping(anndef);
		//---------------------------
		//compute r_k^T r_k
		computeResidueDotProduct(anndef, &residueDotProductResult);
		//printf("r_k^T r_k is %lf %lf \n",residueDotProductResult[0],residueDotProductResult[1]);
		//compute p_k^T A p_k
		computeSearchDirMatrixProduct(anndef,&searchVecMatrixVecProductResult);
		alpha = residueDotProductResult/searchVecMatrixVecProductResult;

		updatedelParameters(alpha);
		updateResidue(anndef);
		printf("residue norm and pkApk  %lf %lf \n",residueDotProductResult,searchVecMatrixVecProductResult);
		prevresidueDotProductResult = residueDotProductResult;
		//compute r_(k+1)^T r_(k+1)
		computeResidueDotProduct(anndef, &residueDotProductResult);
		beta = residueDotProductResult/prevresidueDotProductResult;
		//compute p_(k+1) = r_k+1 + beta p_k
		updateParameterDirection(anndef,beta);
		numberofRuns+=1;
		int i;
		printf("alpha  and beta  is %lf %lf \n", alpha,beta);
		
	}
}

void TrainDNNHF(){
	int currentEpochIdx;
	double learningrate;
	double min_validation_error ;
	
	currentEpochIdx = 0;
	learningrate = 0;
	min_validation_error = 1;

	modelSetInfo = malloc (sizeof(MSI));
	modelSetInfo->crtVal = 0;
	modelSetInfo->bestValue = DBL_MAX;
	modelSetInfo->prevCrtVal = 0;

	//array to store the  training errors and validation error
	double * log_likelihood  = malloc(sizeof(double)*maxEpochNum);
	double * zero_one_error =  malloc(sizeof(double)*maxEpochNum);
	//compute negative-loglikelihood of training data
	printf("computing the mean negative log-likelihood of training data \n");
	loadDataintoANN(inputData,labelMat);
	fwdPassOfANN(anndef);
	log_likelihood[currentEpochIdx] = computeLogLikelihood(anndef->layerList[numLayers-1]->feaElem->yfeatMat,BATCHSAMPLES,anndef->layerList[numLayers-1]->dim,labels);
	modelSetInfo->crtVal = log_likelihood[currentEpochIdx];
	if (modelSetInfo->crtVal < modelSetInfo->bestValue){
		modelSetInfo->bestValue = modelSetInfo->crtVal;
	}
	
	//with the initialisation of weights,checking how well DNN performs on validation data
	printf("computing error on validation data\n");
	setBatchSize(validationDataSetSize);
	reinitLayerFeaMatrices(anndef);
	//load  entire batch into neuralNet
	loadDataintoANN(validationData,validationLabelIdx);
	fwdPassOfANN(anndef);
	zero_one_error[currentEpochIdx] = updatateAcc(validationLabelIdx, anndef->layerList[numLayers-1],BATCHSAMPLES);
	min_validation_error = zero_one_error[currentEpochIdx];
	printf("validation error %lf \n ",zero_one_error[currentEpochIdx] );
		
	initialiseParameterCaches(anndef);
	while(currentEpochIdx <  maxEpochNum){
		printf("epoc number %d \n", currentEpochIdx);
		updateLearningRate(currentEpochIdx,&learningrate);
		//load training data into the ANN and perform forward pass
		setBatchSize(trainingDataSetSize);
		reinitLayerFeaMatrices(anndef);
		loadDataintoANN(inputData,labelMat);
		printf("computing gradient at epoch %d\n", currentEpochIdx);
		fwdPassOfANN(anndef);
		backPropBatch(anndef,FALSE);
		accumulateGradientsofANN(anndef);
		printf("successfully accumulated Gradients \n");
		if (currentEpochIdx == 0){
			runConjugateGradient(TRUE);
		}else{
			runConjugateGradient(FALSE);
		}
		//exit(0);
		printf("successfully completed a run of CG \n");
		updateNeuralNetParamsHF(anndef);
		printf("\n\n\n");
		printLayerMatrices(anndef);
		fwdPassOfANN(anndef);
		log_likelihood[currentEpochIdx] = computeLogLikelihood(anndef->layerList[numLayers-1]->feaElem->yfeatMat,BATCHSAMPLES,anndef->layerList[numLayers-1]->dim,labels);	
		modelSetInfo->prevCrtVal = modelSetInfo->crtVal;
		modelSetInfo->crtVal = log_likelihood[currentEpochIdx];
		printf("log_likelihood %lf \n ",modelSetInfo->crtVal );
		
		printf("checking the performance of the neural net on the validation data\n");
		setBatchSize(validationDataSetSize);
		reinitLayerFeaMatrices(anndef);
		//perform forward pass on validation data and check the performance of the DNN on the validation dat set
		loadDataintoANN(validationData,validationLabelIdx);
		fwdPassOfANN(anndef);
		zero_one_error[currentEpochIdx] = updatateAcc(validationLabelIdx, anndef->layerList[numLayers-1],BATCHSAMPLES);
		if (modelSetInfo->crtVal < modelSetInfo->bestValue){
			modelSetInfo->bestValue = modelSetInfo->crtVal;
			if (zero_one_error[currentEpochIdx]<min_validation_error){
				min_validation_error = zero_one_error[currentEpochIdx];
				cacheParameters(anndef);
			}
			
		}	
		currentEpochIdx+=1;
		

	}
	printf("TRAINING ERROR >>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
	printVector(log_likelihood,maxEpochNum);
	printf("THE  VALIDATION RESULTS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
	printVector(zero_one_error,maxEpochNum);
	printf("The minimum error on the validation data set is %lf percent  and min log_likelihood is %lf \n",min_validation_error*100, modelSetInfo->bestValue);
	free(zero_one_error);
	free(log_likelihood);

}
//----------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------

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
	free(labelMat);
	free(validationData);
	free(validationLabelIdx);
	printf("Finished freeing memory\n");
}

void printLayerDWMatrices(ADLink anndef){
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
}
void printLayerMatrices(ADLink anndef){
	int i; 
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		printf("LAYER ID %d >>>>>>\n",i);
		layer = anndef->layerList[i];
		printf("printing   W >>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
		printMatrix(layer->weights,layer->dim,layer->srcDim);
		printf("printing  B >>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
		printMatrix(layer->weights,1,layer->dim);
		

	}	
}

void printVector(double * vector , int dim){
	int i ;
	printf("[ ");
	for (i = 0; i < dim; i++){
		printf( " %lf ",vector[i]);
	}	
	printf("]\n ");
}

void printMatrix(double * matrix,int row,int col){
int k,j;
int r = row;
int c = col;

for (j = 0 ; j< r;j++){
	for (k =0 ; k <c;k++){
		printf(" %lf ", matrix[j*col +k]);
		if (isnan(matrix[j*row +k])) exit(0);
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

void UnitTest_computeGradientDotProd(){
	int i,k,j; 
	LELink layer;
	double sum =0;
	for ( i  =0 ; i < anndef->layerNum; i++){
		layer = anndef->layerList[i];
		sum += cblas_ddot(layer->dim*layer->srcDim,layer->traininfo->updatedWeightMat, 1, layer->traininfo->updatedWeightMat,1);

	}
	printf("dotproduct of cache gradient vector is %lf \n",sum);
	

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


	

	freeMemoryfromANN();

}


//==============================================================================

int main(int argc, char *argv[]){
	
	  


	/**testing gauss newton product**/
	//if (argc != 11 && argc != 13 ){
	//	printf("The program expects a minimum of  5 args and a maximum of 6 args : Eg : -C config \n -S traindatafile \n -L traininglabels \n -v validationdata \n -vl validationdataLabels \n optional argument : -T testData \n ");
	//}
	//parseCMDargs(argc, argv);
	
	//checking the implementation of doing products with the gauss newton matrix
	
	double  vweightsH[] ={0.775812455441121,0.598968296474700,0.557221023059154,0.299707632069582,0.547657254432423,0.991829829468103,0.483412970471810,0.684773696180478,0.480016831195868,0.746520774837809,0.211422430258003,0.248486726468514,0.0978390795725171,0.708670434511610,0.855745919278928,0.789034249923306,0.742842549759275,0.104522115223072,0.520874909265143,0.846632465315532,0.843613150651208,0.377747297699522,0.272095361461968,0.125303924302938,0.691352539306520,0.555131164040551,0.00847423194155961,0.416031807168918,0.439118205411470,0.784360645016731,0.829997491112413,0.589606438454759,0.142074038271687,0.593313383711502,0.726192799270104,0.428380358133696,0.210792055986133,0.265384268404268,0.993183755212665,0.480756369506705,0.827750131864470,0.603238265822881,0.817841681068066,0.955547170535556,0.650378193390669,0.627647346270778,0.646563331304311,0.0621331286917197,0.938196645878781,0.898716437594415,0.694859162934643,0.310406171229422,0.343545605630366,0.0762294431350499,0.519836940596257,0.608777321469077,0.727159960434458,7.39335770538752e-05,0.169766268796397,0.463291023631989,0.361852151271785,0.952065614552366,0.932193359196872,0.958068660798924,0.206538420408243,0.159700734052653,0.571760567030289,0.592919191548301,0.698610749870688,0.305813464081036,0.393851341078336,0.336885826085107,0.128993041260436,0.0869363561072062,0.548943348806040,0.317733245140830,0.992747576055697,0.723565253441235,0.572101390105790,0.717227171632155,0.976559227688336,0.426990558230810,0.913013771059024,0.897311254281181,0.835228567475913,0.0479581714127454,0.359280569413193,0.295753896108793,0.629125618231216,0.136183801688937,0.374045801279154,0.864781698209823,0.250273591941488,0.0295055785950556,0.999448971390378,0.605142675082282,0.244195121925386,0.438195271553610,0.735093567145817,0.557989092238218};	
	double vbaisH[] ={0,0,0,0,0,0,0,0,0,0};

	double  vweights0[] ={0.775812455441121,0.598968296474700,0.557221023059154,0.299707632069582,0.547657254432423,0.991829829468103,0.483412970471810,0.684773696180478,0.480016831195868,0.746520774837809,0.211422430258003,0.248486726468514,0.0978390795725171,0.708670434511610,0.855745919278928,0.789034249923306,0.742842549759275,0.104522115223072,0.520874909265143,0.846632465315532,0.843613150651208,0.377747297699522,0.272095361461968,0.125303924302938,0.691352539306520,0.555131164040551,0.00847423194155961,0.416031807168918,0.439118205411470,0.784360645016731,0.829997491112413,0.589606438454759,0.142074038271687,0.593313383711502,0.726192799270104,0.428380358133696,0.210792055986133,0.265384268404268,0.993183755212665,0.480756369506705,0.827750131864470,0.603238265822881,0.817841681068066,0.955547170535556,0.650378193390669,0.627647346270778,0.646563331304311,0.0621331286917197,0.938196645878781,0.898716437594415,0.694859162934643,0.310406171229422,0.343545605630366,0.0762294431350499,0.519836940596257,0.608777321469077,0.727159960434458,7.39335770538752e-05,0.169766268796397,0.463291023631989,0.361852151271785,0.952065614552366,0.932193359196872,0.958068660798924,0.206538420408243,0.159700734052653,0.571760567030289,0.592919191548301,0.698610749870688,0.305813464081036,0.393851341078336,0.336885826085107,0.128993041260436,0.0869363561072062,0.548943348806040,0.317733245140830,0.992747576055697,0.723565253441235,0.572101390105790,0.717227171632155,0.976559227688336,0.426990558230810,0.913013771059024,0.897311254281181,0.835228567475913,0.0479581714127454,0.359280569413193,0.295753896108793,0.629125618231216,0.136183801688937,0.374045801279154,0.864781698209823,0.250273591941488,0.0295055785950556,0.999448971390378,0.605142675082282,0.244195121925386,0.438195271553610,0.735093567145817,0.557989092238218};	
	double vbaisO[] ={0,0,0,0,0,0,0,0,0,0};

	double data[] ={0.254790156597005,0.224040030824219,0.667832727013717,0.844392156527205,0.344462411301042,0.780519652731358,0.675332065747000,0.00671531431847749,0.602170487581795,0.386771194520985,0.915991244131425,0.00115105712910724,0.462449159242329,0.424349039815375,0.460916366028964,0.770159728608609,0.322471807186779,0.784739294760742,0.471357153710612,0.0357627332691179};
	

	double lab[] ={4,1};
	double labmat[] ={0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0};
	
	numLayers = 2;
	hidUnitsPerLayer = malloc(sizeof(int)*numLayers);
	actfunLists = malloc( sizeof(ActFunKind)*numLayers);
	hidUnitsPerLayer[0]=10;
	actfunLists[0]= TANH;
	actfunLists[1]= SOFTMAX;		
	target=CLASSIFICATION;
	BATCHSAMPLES= 2;
	errfunc = XENT;
	inputDim = 10;
	targetDim =10;
	doHF =TRUE;
	useGNMatrix =TRUE;
	maxNumOfCGruns =10;
	
	

	initialiseDNN();
	anndef->layerList[0]->feaElem->xfeatMat = data;
	anndef->labelMat = labmat;

	double Weight [] ={-0.33792351, 0.13376346, -0.06821584,0.31259467,0.30669813, -0.24911232,-0.24487114,0.3306844, 0.50186652,0.41181357,-0.15575338,0.00109011,0.20097358,0.2330034,-0.14213318,0.06703706, 0.00337744, -0.53263998,0.29886659,0.41916242,-0.14800999,0.12641018, -0.46514654, -0.1436961, 0.47448121,0.16582645,-0.11260893,0.31628802, -0.20064598,0.07459834, 0.4043588,-0.06991851,0.33098616, -0.39023389,0.22375668,0.22410759,-0.30804781,0.46541917 ,-0.06338163,0.44838317,-0.48220484, -0.34584617, -0.49584745,0.19157248,0.10365625,0.03648946,-0.50026342,0.06729657, -0.18658887,0.00325,-0.42514847,0.11742482,0.07223874, -0.5403129, 0.12865095,0.451458, 0.31825324,0.53904824,0.50259215,0.31983069,-0.23524579,0.13683939, -0.02399704, -0.33337114, -0.12891477, -0.48870689,-0.05296651,0.52800974, -0.41195013, -0.41694734, 0.26128892,0.09563634, -0.031075, -0.43037101, -0.2966262, 0.4381399,-0.09119193,0.03927353, -0.54092147, -0.21838607,-0.06913007,0.12285307,0.45811304,0.13773762,0.22565903,-0.38358795,0.26954896,0.36259999,0.14648924, -0.06757814,-0.38058746,0.07493898,0.03091815,0.49451543, -0.02151544,0.00280386, 0.04039804,0.34966835, -0.48515551,0.18559222};


	printf("input data >>>>>\n");
	printMatrix(anndef->layerList[0]->feaElem->xfeatMat,BATCHSAMPLES,anndef->layerList[0]->dim);
	

    copyMatrixOrVec(Weight,anndef->layerList[0]->weights,anndef->layerList[0]->dim*anndef->layerList[0]->srcDim);
	copyMatrixOrVec(Weight,anndef->layerList[1]->weights,anndef->layerList[1]->dim*anndef->layerList[1]->srcDim);
	
	printf("PRINTING WEIGHTS OF LAYERS>>>>>>>>>>>>>>>\n");
	printf("Layer 0 weights \n");
	printMatrix(anndef->layerList[0]->weights,anndef->layerList[0]->dim,anndef->layerList[0]->srcDim);
	printf("Layer 0 bias \n");
	printMatrix(anndef->layerList[0]->bias,anndef->layerList[0]->dim,1);

	printf("Layer 0 weights \n");
	printMatrix(anndef->layerList[1]->weights,anndef->layerList[1]->dim,anndef->layerList[1]->srcDim);
	printf("Layer 0 bias \n");
	printMatrix(anndef->layerList[1]->bias,anndef->layerList[1]->dim,1);
	printf("done  WEIGHTS OF LAYERS>>>>>>>>>>>>>>>\n");

	//fwdPassOfANN(anndef);
	

	/*
	copyMatrixOrVec(vweightsH,anndef->layerList[0]->gnInfo->vweights,inputDim*anndef->layerList[0]->dim);
	printf("vweights of hidden layer\n");
	printMatrix(anndef->layerList[0]->gnInfo->vweights,10,10);

	copyMatrixOrVec(vbaisH,anndef->layerList[0]->gnInfo->vbiases,anndef->layerList[0]->dim);
	printf("vbais of hidden layer\n");
	printMatrix(anndef->layerList[0]->gnInfo->vbiases,1,10);

	copyMatrixOrVec(vweights0,anndef->layerList[1]->gnInfo->vweights,anndef->layerList[1]->dim*anndef->layerList[1]->srcDim);
	printf("vweights of output layer\n");
	printMatrix(anndef->layerList[1]->gnInfo->vweights,10,10);

	copyMatrixOrVec(vbaisO,anndef->layerList[1]->gnInfo->vbiases,anndef->layerList[1]->dim);
	printf("vbias of hidden layer\n");
	printMatrix(anndef->layerList[1]->gnInfo->vbiases,1,10);
	*/
	printf("computing and printing JV>>>>>>>>>>>>>>>>>\n");
	fwdPassOfANN(anndef);
	//double loglik =computeLogLikelihood(anndef->layerList[numLayers-1]->feaElem->yfeatMat,BATCHSAMPLES,anndef->layerList[numLayers-1]->dim,lab);	
	//printf("log like %lf\n>>>>",loglik);
	printf("Computing derivative  >>>>>>>>>>>>>>>\n");
	
	backPropBatch(anndef,FALSE);
	accumulateGradientsofANN(anndef);
	computeNormOfGradient(anndef);
	computeNormOfAccuGradient(anndef);

	runConjugateGradient(TRUE);

		
	/*
	printf("PRINTING JV >>>>>>>>>>>>>>>>>>>>>>\n");
	printf("vbias of hidden layer\n");
	printMatrix(anndef->layerList[1]->gnInfo->Ractivations,BATCHSAMPLES,anndef->layerList[1]->dim);


	printf("PRINTING HJV >>>>>>>>>>>>>>>>>>>>>>\n");	
	printMatrix(anndef->layerList[1]->feaElem->yfeatMat,BATCHSAMPLES,anndef->layerList[1]->dim);


	printf("de/dw of hidden layer\n");
	printMatrix(anndef->layerList[0]->traininfo->dwFeatMat,anndef->layerList[0]->dim,anndef->layerList[0]->srcDim);

	printf("de/db of hidden layer\n");
	printMatrix(anndef->layerList[0]->traininfo->dbFeaMat,1,10);

	printf("de/dw of output layer\n");
	printMatrix(anndef->layerList[1]->traininfo->dwFeatMat,anndef->layerList[1]->dim,anndef->layerList[1]->srcDim);

	printf("de/db of output layer\n");
	printMatrix(anndef->layerList[1]->traininfo->dbFeaMat,1,10);

	*/




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
double a [] ={ 1,2 , 3 ,3 ,1};
printf("DOT Produve result %lf \n",cblas_ddot(5,a,1,a,1));
printf("TEST is %lf \n",exp(-10000));

  //TrainDNNGD();
  weightdecay =1; 
	
	//TrainDNNHF();
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

