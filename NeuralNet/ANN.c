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

/*training data set and validation data set*/
static int BATCHSAMPLES; //the number of samples to load into the DNN
static double * inputData;
static double * labels;
static double * validationData;
static double * validationLabelIdx;
static int trainingDataSetSize;
static int validationDataSetSize;


/*configurations for DNN architecture*/
static Boolean doHF;
static Boolean useGNMatrix;
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

	fp = fopen(filepath,"r");
	i = 0;
	while(getline(&line,&len,fp)!=-1){
		cleanString(line);
		//extracting labels 
		if (target==CLASSIFICATION){
			id  = (int) strtod(line,NULL);
			for (c = 0; c < targetDim;c++){
				labelMat[i*targetDim +c] = 0;
			}
			labelMat[i*targetDim + id] = 1;
		}
		//extracting function outputs
		else{
			id  = strtod(line,NULL);
			labelMat[i] = id ;
		}
		if (strcmp(datatype,"train")==0){
			if (i*targetDim > trainingDataSetSize*targetDim){
				printf("Error! : the number of training labels doesnt match the size of the training set \n");
				exit(0);
			}
		}else if(strcmp(datatype,"validation")==0){
			if(i*targetDim > validationDataSetSize*targetDim){
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
		token = strtok(line," ");
		while (token != NULL){
			matrix[i] = strtod(token,NULL);
			token = strtok(NULL," ");
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
			validationLabelIdx = malloc(sizeof(double)*(validationDataSetSize*targetDim));
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
	return (rand()+1.0)/(RAND_MAX+1.0)* (2*limit +1) +limit;
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
		/*standard intialisation : setting values according to fan-in*/
		//randm = random_normal();
		//biasVec[i] = randm*(1/sqrt(srcDim));

		/* bengio;s proposal for a new type of initialisation to ensure 
			the variance of error derivatives are comparable accross layers*/
		biasVec[i] = genrandWeight(sqrt(6)/sqrt(dim+srcDim+1));
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
			/*standard intialisation : setting values according to fan-in*/
			//randm = random_normal();
			//*weights = randm * 1/(sqrt(srcDim));
			
			/* bengio;s proposal for a new tpye of initialisation to ensure 
			the variance of error derivatives are comparable accross layers*/
			*weights = genrandWeight(sqrt(6)/sqrt(dim+srcDim+1));
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
	
	//intialise traininfo and allocating extra memory for setting hooks
	layer->traininfo = (TRLink) malloc(sizeof(TrainInfo) + sizeof(double)*(numOfElems*2));
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
double computeTanh(double x){
	return 2*(computeSigmoid(2*x))-1;
}
double computeSigmoid(double x){
	double result;
	result = 1/(1+ exp(-1*x));
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
					for (i = 0;i < BATCHSAMPLES;i++){
						sum = 0;
						for (j =0; j<layer->dim;j++){
							layer->feaElem->yfeatMat[i*layer->dim+j] = exp(layer->feaElem->yfeatMat[i*layer->dim+j]);
							sum+=layer->feaElem->yfeatMat[i*layer->dim+j];
						}
						for (j =0; j<layer->dim;j++){
							layer->feaElem->yfeatMat[i*layer->dim+j]/=sum ;
						}
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
					printf("Reaches here 1\n");
					computeLossHessSoftMax(outlayer);
				default:
					break;
			}
		case SSE :
			break;
	}
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
		#endif
	}
}

//----------------------------------------------------------------------------------------------------------
/**this segment of the code is reponsible for accumulating the gradients **/
//---------------------------------------------------------------------------------------------------------
void setHook(Ptr m, Ptr ptr,int incr){
	Ptr *p;
	printf("hello\n");
   p = (Ptr *) m; 
   printf("hello casting success\n");
   p -= incr; *p = ptr;
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
void setParameterDirections(double *weights, double* bias, LELink layer){
	assert(layer->gnInfo !=NULL);
	#ifdef CBLAS
	cblas_dcopy(layer->dim*layer->srcDim,weights,1,layer->gnInfo->vweights,1);
	cblas_dcopy(layer->dim,bias,1,layer->gnInfo->vbiases,1);
	#else
	/**CPU Version**/
	int i;
	for (i = 0; i<layer->dim*layer->srcDim;i++){
		layer->gnInfo->vweights[i] = weights[i];
	}
	for (i = 0; i<layer->dim;i++){
		layer->gnInfo->vbiases[i] = bias[i];
	}	
	#endif
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
	#ifdef CBLAS
		int i,off;
		for (i = 0, off = 0; i < BATCHSAMPLES;i++, off += layer->dim){
			 cblas_dcopy(layer->dim, layer->gnInfo->vbiases, 1, layer->gnInfo->Ractivations + off, 1);
		}
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

//----------------------------------------------------------------------------------------------------
/*This section deals with running schedulers to iteratively update the parameters of the neural net**/
//-------------------------------------------------------------------------------------------------------

/**void resetWeights(ADLink anndef){
	int i;
	LELink layer;
	for( i = 0; i < anndef->layerNum;i++){
		layer = anndef->layerList[i];
		double* weightCache = (double *) getHook(layer->traininfo,1);
		double* biasCache = (double *) getHook(layer->traininfo,2);
		printf("size of retrieved memory %lu %lu \n",sizeof(weightCache),sizeof(biasCache));
		#ifdef CBLAS
			cblas_dcopy(layer->dim*layer->srcDim, layer->weights, 1, weightCache, 1);
			cblas_dcopy(layer->dim, layer->bias, 1, biasCache, 1);
		#endif
	}
}**/



void fillCache(LELink layer,int dim,Boolean weights){
	#ifdef CBLAS
	if (weights){
		double* paramCache = (double *) getHook(layer->traininfo,1);
		cblas_dcopy(dim,layer->weights, 1,paramCache,1);
	}else{
		double* paramCache = (double *) getHook(layer->traininfo,2);
		cblas_dcopy(dim,layer->bias, 1,paramCache,1);
	}
	#endif
}

void cacheParameters(ADLink anndef){
	int i;
	LELink layer;
	for (i =(anndef->layerNum-1) ; i >=0; --i){
		layer = anndef->layerList[i];
		fillCache(layer,layer->dim*layer->srcDim,TRUE);
		fillCache(layer,layer->dim,FALSE);
	}	
}

void intialiseParameterCaches(ADLink anndef){
	int i;
	LELink layer;
	for (i =(anndef->layerNum-1) ; i >=0; --i){
		layer = anndef->layerList[i];
		double* weightCache = malloc(sizeof(double)*(layer->dim*layer->srcDim));
		double* biasCache =  malloc(sizeof(double)*layer->dim);
		setHook(layer->traininfo,weightCache,1);
		setHook(layer->traininfo,biasCache,2);
	}	
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
/** the function calculates the percentage of the data samples correctly labelled by the DNN*/
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
			if (predictions[i] != labels[i]){
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
void updateNeuralNetParams(ADLink anndef, double lrnrate, double momentum, double weightdecay){
	int i;
	LELink layer;
	for (i =(anndef->layerNum-1) ; i >=0; --i){
		layer = anndef->layerList[i];
		//if we have a regularised error function: 
		if (weightdecay > 0){
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
	if( (-1*lrnrate) < minLR)return FALSE;
	if (fabs(lrnrate-0)<0.0001) return FALSE;
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
	intialiseParameterCaches(anndef);
	if(modelSetInfo->crtVal<modelSetInfo->bestValue){
		cacheParameters(anndef);
		modelSetInfo->bestValue = modelSetInfo->crtVal;
		printf("successfully updated weight caches\n");
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
		updatateAcc(validationLabelIdx,anndef->layerList[numLayers-1],validationDataSetSize);
		if (modelSetInfo->crtVal < modelSetInfo->bestValue){
			cacheParameters(anndef);
			modelSetInfo->bestValue = modelSetInfo->crtVal;
		}
		currentEpochIdx+=1;
	}
	printf("The minimum error on the validation data set is %lf percent \n",modelSetInfo->bestValue*100);
}

//--------------------------------------------------------------------------------------------------------
/**This section of the code implements HF full batch training**/


void initialiseResidueaAndSearchDirection(ADLink aandef){
	int i; 
	LELink layer;
	for (i = 0; i<anndef->layerNum;i++){
		layer = anndef->layerList[i];
		#ifdef CBLAS
		//ro = A*0 +b 
		cblas_dcopy(layer->dim*layer->srcDim, layer->traininfo->updatedWeightMat, 1,layer->cgInfo->residueUpdateWeights, 1);
		cblas_dcopy(layer->dim*layer->srcDim, layer->traininfo->updatedWeightMat, 1,layer->cgInfo->searchDirectionUpdateWeights, 1);
				
		cblas_dcopy(layer->dim,layer->traininfo->updatedBiasMat,1,layer->cgInfo->residueUpdateBias,1);
		cblas_dcopy(layer->dim,layer->traininfo->updatedBiasMat,1,layer->cgInfo->searchDirectionUpdateBias,1);

		//po =-ro
		cblas_dscal(layer->dim*layer->srcDim,-1,layer->cgInfo->searchDirectionUpdateWeights,1);
		cblas_dscal(layer->dim,-1,layer->cgInfo->searchDirectionUpdateBias,1);
		#endif
	}

}

void runConjugateGradient(Boolean firstEverRun){
	if (firstEverRun){
		initialiseResidueaAndSearchDirection(anndef);
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
	intialiseParameterCaches(anndef);
	if(modelSetInfo->crtVal<modelSetInfo->bestValue){
		cacheParameters(anndef);
		modelSetInfo->bestValue = modelSetInfo->crtVal;
		printf("successfully updated weight caches\n");
	}
	
	while(terminateSchedNotTrue(currentEpochIdx,learningrate)){
		/*for each iteration: compute and  accumulate the gradient and then run CG*/
		setBatchSize(trainingDataSetSize);
		reinitLayerFeaMatrices(anndef);
		loadDataintoANN(inputData,labels);
		fwdPassOfANN(anndef);
		// run backpropagation and compute gradients
		backPropBatch(anndef,FALSE);
		accumulateGradientsofANN(anndef);
		if (currentEpochIdx == 0){
			runConjugateGradient(TRUE);
		}else{
			runConjugateGradient(FALSE);
		}


	}

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


//=================================================================================

int main(int argc, char *argv[]){
	/**testing gauss newton product**/
	
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
	double labels[] ={1.2,1.2};


	/**testting cpoy*/
	double *T = malloc(sizeof(double)*2);
	cblas_dcopy(2,labels, 1, T, 1);
	printf("Copied values are %lf  %lf \n",*T,*(T+1));
	free(T);	


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
}  

