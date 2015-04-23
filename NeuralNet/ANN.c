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
static int trainingDataSetSize;
static int validationDataSetSize;

/*configurations for DNN architecture*/
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

	fp = fopen(filepath,"r");
	while(getline(&line,&len,fp)!=-1){
		token = strtok(line," : ");
		if (strcmp(token,"momentum")==0){
			token = strtok(NULL," : ");
			momentum = strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"weightdecay")==0){
			token = strtok(NULL," : ");
			weightdecay = strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;	
		}
		if (strcmp(token,"minLR")==0){
			token = strtok(NULL," : ");
			minLR = strtod (token,NULL);
			token = strtok(NULL," : ");
			continue	;
		}
		if (strcmp(token,"maxEpochNum")==0){
			token = strtok(NULL," : ");
			maxEpochNum = (int) strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"initLR")==0){
			token = strtok(NULL," : ");
			initLR =  strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"threshold")==0){
			token = strtok(NULL," : ");
			threshold = strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"numLayers")==0){
			token = strtok(NULL," : ");
			numLayers = (int) strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"inputDim")==0){
			token = strtok(NULL," : ");
			inputDim = (int) strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"targetDim")==0){
			token = strtok(NULL," : ");
			targetDim = (int) strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"trainingDataSetSize")==0){
			token = strtok(NULL," : ");
			trainingDataSetSize = (int) strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"validationDataSetSize")==0){
			token = strtok(NULL," : ");
			validationDataSetSize = (int)strtod (token,NULL);
			token = strtok(NULL," : ");
			continue;
		}
		if (strcmp(token,"Errfunc")==0){
			token = strtok(NULL," : ");
			if (strcmp("XENT",token)==0){
				errfunc = XENT;
				target = CLASSIFICATION;
			}else if (strcmp("SSE",token)==0){
				errfunc = SSE ;
				target = REGRESSION ;
			}	
			continue;
		}
		
		if (strcmp(token,"hiddenUnitsPerLayer")==0){
			hidUnitsPerLayer = malloc(sizeof(int)*numLayers);
			token = strtok(NULL," : ");
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
	if (strcmp(argv[2],"-C")!=0){
		printf("the first argument to ANN must be the config file \n");
		exit(0);
	}
	for (i = 2 ; i < argc;i++){
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

//-----------------------------------------------------------------------------------------------------------
/**this section of the src code deals with initialisation of ANN **/
//-----------------------------------------------------------------------------------------------------------
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

		/* bengio;s proposal for a new tpye of initialisation to ensure 
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
	//intialise traininfo
	layer->info = (TRLink) malloc(sizeof(TrainInfo));
	assert(layer->info!= NULL);
	layer->info->dwFeatMat = malloc(sizeof(double)*numOfElems);
	layer->info->dbFeaMat = malloc(sizeof(double)*layer->dim);
	layer->info->updatedWeightMat = NULL;
	layer->info->updatedBiasMat = NULL;

	if (momentum > 0) {
		layer->info->updatedWeightMat = malloc(sizeof(double)*numOfElems);
		layer->info->updatedBiasMat = malloc(sizeof(double)*(layer->dim));
		initialiseWithZero(layer->info->updatedWeightMat,numOfElems);
		initialiseWithZero(layer->info->updatedBiasMat,layer->dim);
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

/**load entire batch into the neural net**/
void loadDataintoANN(double *samples, double *labels){
	anndef->layerList[0]->feaElem->xfeatMat = samples;
	anndef->labelMat = labels;
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
		free(ones);
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
				default:
					break;
			}
		case (SSE):
			subtractMatrix(layer->errElem->dyFeatMat,anndef->labelMat,layer->dim*BATCHSAMPLES);	
			break;
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
	int i, dim;
	double accCount,holdingVal;
	accCount=0;
	/**classification**/
	if (anndef->target==CLASSIFICATION){
		double *predictions = malloc(sizeof(double)*dataSize);
		if (layer->dim >1){
			dim = layer->dim;
			findMaxElement(layer->feaElem->yfeatMat,dataSize,dim,predictions);
		}else{
			perfBinClassf(layer->feaElem->yfeatMat,predictions,dataSize);
		}
		
		for (i = 0; i<dataSize;i++){
			if (predictions[i] == labels[i]){
				accCount+=1;
			}	
		}
		free(predictions);
	}
	/**regression**/
	else{
		subtractMatrix(layer->feaElem->yfeatMat, labels, dataSize);
		for (i = 0;i<dataSize*layer->dim;i++){
			holdingVal = layer->feaElem->yfeatMat[i];
			accCount+= holdingVal*holdingVal;
		}
	}		
		
	modelSetInfo->crtVal = accCount/dataSize;
	printf("The critical value is %lf \n", modelSetInfo->crtVal);
}

/* this function allows the addition of  two matrices or two vectors*/
void addMatrixOrVec(double *weights, double *dwFeatMat,int dim){
	//blas routine
	#ifdef CBLAS
		cblas_daxpy(dim,1,weights,1,dwFeatMat,1);
	#else
		int i;
		for (i =0;i<dim;i++){
			dwFeatMat[i] = dwFeatMat[i] + weights[i];
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
			scaleMatrixOrVec(layer->info->dwFeatMat,-1*lrnrate,layer->dim*layer->srcDim);
			scaleMatrixOrVec(layer->info->dbFeaMat,-1*lrnrate,layer->dim);
		}
		if (momentum > 0){
			scaleMatrixOrVec(layer->info->updatedWeightMat,momentum,layer->dim*layer->srcDim);
			scaleMatrixOrVec(layer->info->updatedBiasMat,momentum,layer->dim);
			addMatrixOrVec(layer->info->dwFeatMat,layer->info->updatedWeightMat,layer->dim*layer->srcDim);
			addMatrixOrVec(layer->info->dbFeaMat,layer->info->updatedBiasMat,layer->dim);
			//updating parameters: first we need to descale the lambda from weights and bias
			if (weightdecay > 0){
			scaleMatrixOrVec(layer->weights,1/weightdecay,layer->dim*layer->srcDim);
			scaleMatrixOrVec(layer->bias,1/weightdecay,layer->dim);
			}
			addMatrixOrVec(layer->info->updatedWeightMat,layer->weights,layer->dim*layer->srcDim);
			addMatrixOrVec(layer->info->updatedBiasMat,layer->bias,layer->dim);
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
		*lrnrate =  initLR;
	}else if (modelSetInfo !=NULL){
		crtvaldiff = modelSetInfo->crtVal - modelSetInfo->prevCrtVal;
		if (target==CLASSIFICATION){
			if (crtvaldiff < threshold){
				*lrnrate /=2;
				printf("Learning rate has been halved !! \n");
			}
		}else if(crtvaldiff >0){
			*lrnrate /=2;
			printf("Learning rate has been halved !! \n");
		}
		
	}
}

Boolean terminateSchedNotTrue(int currentEpochIdx,double lrnrate,MSI* modelSetInfo){
	printf("lrn rate %f\n",lrnrate);
	if (currentEpochIdx == 0) return TRUE;
	if (currentEpochIdx >=0 && currentEpochIdx >= maxEpochNum)return FALSE;
	if(lrnrate< minLR)return FALSE;
	if(target==REGRESSION){
		if ((modelSetInfo->crtVal/modelSetInfo->prevCrtVal)>=0.98 && currentEpochIdx>=10){
		return FALSE;
		}
	}
	return TRUE; 
}

void TrainDNN(){
	int currentEpochIdx;
	double learningrate;
	
	currentEpochIdx = 0;
	learningrate = 0;
	modelSetInfo = malloc (sizeof(MSI));
	modelSetInfo->crtVal = 0;

	//with the initialisation of weights,check how well DNN performs on validation data
	setBatchSize(validationDataSetSize);
	reinitLayerFeaMatrices(anndef);
	//load  entire batch into neuralNet
	loadDataintoANN(validationData,validationLabelIdx);
	fwdPassOfANN(anndef);
	printf("successfully performed forward pass of DNN on validation data\n");
	updatateAcc(validationLabelIdx, anndef->layerList[numLayers-1],BATCHSAMPLES);
	printf("successfully accumulated counts \n");
	
	while(terminateSchedNotTrue(currentEpochIdx,learningrate,modelSetInfo)){
		printf("epoc number %d \n", currentEpochIdx);
		updateLearningRate(currentEpochIdx,&learningrate);
		//load training data into the ANN and perform forward pass
		setBatchSize(trainingDataSetSize);
		reinitLayerFeaMatrices(anndef);
		loadDataintoANN(inputData,labels);
		fwdPassOfANN(anndef);
		
		// run backpropagation and update the parameters:
		BackPropBatch(anndef);
		updateNeuralNetParams(anndef,learningrate,momentum,weightdecay);
		
		//forward pass of DNN on validation data if VD is provided
		if (validationData != NULL && validationLabelIdx != NULL){
			setBatchSize(validationDataSetSize);
			reinitLayerFeaMatrices(anndef);
			//perform forward pass on validation data and check the performance of the DNN on the validation dat set
			loadDataintoANN(validationData,validationLabelIdx);
			fwdPassOfANN(anndef);
			modelSetInfo->prevCrtVal = modelSetInfo->crtVal;
			updatateAcc(validationLabelIdx,anndef->layerList[numLayers-1],validationDataSetSize);
		}else{
			modelSetInfo->prevCrtVal = modelSetInfo->crtVal;
			updatateAcc(labels,anndef->layerList[numLayers-1],trainingDataSetSize);
		}
		currentEpochIdx+=1;
	}
	




}
//========================================================================================================
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
					if(anndef->layerList[i]->info->updatedBiasMat !=NULL){
						free(anndef->layerList[i]->info->updatedBiasMat);
					}
					if (anndef->layerList[i]->info->updatedWeightMat!= NULL){
						free(anndef->layerList[i]->info->updatedWeightMat);
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
	if (argc != 12 && argc != 14 ){
		printf("The program expects a minimum of  5 args and a maximum of 6 args : Eg : -C config \n -S traindatafile \n -L traininglabels \n -v validationdata \n -vl validationdataLabels \n optional argument : -T testData \n ");
	}
	parseCMDargs(argc, argv);
	printf("'REACHES HERE'\n" );


	int i,j,c;
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
 **
	c=0;
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

	 initialise();
	for ( i = 0; i<numLayers ;i++){
		LELink layer = anndef->layerList[i];
		printf("The trained weights of layer %i is \n",i);
		for (j = 0; j<layer->dim;j++){
			for (c =0 ; c< layer->srcDim ;c++){
				printf(" %lf ",layer->weights[j*layer->dim +c] );
			}
			printf("\n");
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

	
	//fwdPassOfANN(anndef);
	TrainDNN();

	for ( i = 0; i<numLayers ;i++){
		LELink layer = anndef->layerList[i];
		printf("The trained weights of layer %i is \n",i);
		for (j = 0; j<layer->dim;j++){
			for (c =0 ; c< layer->srcDim ;c++){
				printf(" %lf ",layer->weights[j*layer->dim +c] );
			}
			printf("\n");
		}
	}

	freeMemoryfromANN();
	//unitTests();
}

