
typedef void * Ptr;
typedef enum {FALSE, TRUE} Boolean;
typedef enum {XENT, SSE} ObjFuncKind;
typedef enum {REGRESSION, CLASSIFICATION} OutFuncKind;
typedef enum {HIDDEN,INPUT,OUTPUT} LayerRole;
typedef enum {SIGMOID,IDENTITY,TANH,SOFTMAX} ActFunKind;


typedef struct _LayerElem *LELink;
typedef struct _ANNdef *ADLink;
typedef struct _FeatElem *FELink;
typedef struct _ErrorElem *ERLink;
typedef struct _TrainInfo *TRLink;
typedef struct _GaussNewtonProductInfo *GNProdInfo;
typedef struct _MSI *MSLink;



/*model set info -this struct is needed to compare the error on the validation dataset between two epochs*/
typedef struct _MSI{
double crtVal;
double prevCrtVal;
}MSI;

/**this struct stores the search directions,residues and dw for each iteration of CG**/
typedef struct _ConjugateGradientInfo{
   double *delweightsUpdate;
   double *delbiasUpdate;
   double *residueUpdateWeights;
   double * residueUpdateBias;
   double *searchDirectionUpdateWeights;
   double * searchDirectionUpdateBias;
 }ConjuageGradientInfo;



/** this struct stores the directional error derivatives with respect to weights and biases**/
typedef struct _GaussNewtonProdInfo{
	double *vweights;
   	double *vbiases;
   	double *Ractivations;
}GaussNewtonProductInfo ;

typedef struct _TrainInfo{
	double *dwFeatMat; /* dE/dw matrix*/
	double *dbFeaMat; /* dE/db  vector */
	double *updatedWeightMat; /* stores the velocity in the weight space*/
	double *updatedBiasMat;/* stores the velocity in the bias space*/
	double *labelMat;
}TrainInfo;

typedef struct _ErrorElem{
	double *dxFeatMat;
	double *dyFeatMat;
}ErrElem;

typedef struct _FeatElem{
	double *xfeatMat;/* is a BatchSample size by feaDim matix */
	double *yfeatMat; /* is BatchSample Size by node matrix*/
}FeaElem;

/*structure for individual layers*/
typedef struct _LayerElem{
	int  id; /* each layer has a unique layer id */
	int  dim ; /* number of units in the hidden layer */
	LayerRole role; /* the type of layer : input,hidden or output */
	ActFunKind actfuncKind; /*each layer is allowed to have its own non-linear activation function */
	LELink src; /* pointer to the input layer */
	int srcDim; /* the number of units in the input layer */
	double *weights;/* the weight matrix of the layer should number of nodes by input dim*/
	double *bias; /* the bias vector */
	FELink feaElem; /* stores the  input activations coming into the layer as well the output activations going out from the layer */
	ERLink errElem;
	TRLink traininfo;/*struct that stores the error derivatives with respect to weights and biases */
	GNProdInfo gnInfo;
}LayerElem;

/*structure for ANN*/
typedef struct _ANNdef{
	int layerNum;
	LELink *layerList;/* list of layers*/
	OutFuncKind target; /* the activation function of the final output layer */
	ObjFuncKind errorfunc; /* the error function of the ANN */
	double *labelMat ; /* the target labels : BatchSample by targetDim matrix*/
}ANNDef;

/**This section of the code deals with parsing Command Line arguments**/

void cleanString(char *Name);
void loadLabels(double *labelMat,char*filepath,char *datatype);
void loadMatrix(double *matrix,char *filepath, char *datatype);
void parseCfg(char * filepath);
void parseCMDargs(int argc, char *argv[]);

/**This section of the code deals with handling the batch sizes of the data**/
void setBatchSize(int sampleSize);

/**this section of the src code deals with initialisation of ANN **/
void setUpForHF(ADLink anndef);
void reinitLayerFeaMatrices(ADLink anndef);
void initialiseErrElems(ADLink anndef);
void initialiseWithZero(double * matrix, int dim);
double genrandWeight(double limit);
double drand();
double random_normal();
void initialiseBias(double *biasVec,int dim, int srcDim);
void initialiseWeights(double *weightMat,int length,int srcDim);
void initialiseLayer(LELink layer,int i, LELink srcLayer);
void  initialiseANN();
void initialise();

/*this section of the code implements the  forward propgation of a deep neural net **/
double computeTanh(double x);
double computeSigmoid(double x);
void computeNonLinearActOfLayer(LELink layer);
void computeLinearActivation(LELink layer);
void loadDataintoANN(double *samples, double *labels);
void fwdPassOfDNN(ADLink anndef);


/*This section of the code implements the back-propation algorithm  to compute the error derivatives**/
void computeDrvAct(double *dyfeat , double *yfeat,int len);
void computeActivationDrv (LELink layer);
void sumColsOfMatrix(double *dyFeatMat,double *dbFeatMat,int dim,int batchsamples);
void subtractMatrix(double *dyfeat, double* labels, int dim);
void calcOutLayerBackwardSignal(LELink layer,ADLink anndef );
void backPropBatch(ADLink anndef);


/*This section deals with running schedulers to iteratively update the parameters of the neural net**/
void perfBinClassf(double *yfeatMat, double *predictions,int dataSize);
void findMaxElement(double *matrix, int row, int col, double *vec);
void updatateAcc(double *labels, LELink layer,int dataSize);
void addMatrixOrVec(double *weightMat, double* dwFeatMat, int dim);
void scaleMatrixOrVec(double* weightMat, double learningrate,int dim);
void updateNeuralNetParams(ADLink anndef, double lrnrate, double momentum, double weightdecay);
void updateLearningRate(int currentEpochIdx, double *lrnRate);
Boolean terminateSchedNotTrue(int currentEpochIdx,double lrnrate);
void TrainDNNGD();


void freeMemoryfromANN();
/*This function is used to check the correctness of implementing the forward pass of DNN and the back-propagtion algorithm*/
void unitTests();
