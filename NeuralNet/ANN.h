
typedef enum {TRUE, FALSE} Boolean;
typedef enum {XENT, SSE} ObjFuncKind;
typedef enum {REGRESSION, CLASSIFICATION} OutFuncKind;
typedef enum {HIDDEN,INPUT,OUTPUT} LayerRole;
typedef enum {SIGMOID,IDENTITY,TANH,SOFTMAX} ActFunKind;

typedef struct _LayerElem *LELink;
typedef struct _ANNdef *ADLink;
typedef struct _FeatElem *FELink;
typedef struct _ErrorElem *ERLink;
typedef struct _TrainInfo *TRLink;
typedef struct _MSI *MSLink;


/*model set info -this struct is needed to compare the error on the validation dataset between two epochs*/
typedef struct _MSI{
double crtVal;
double prevCrtVal;
}MSI

typedef struct _TrainInfo{
	double *dwFeatMat;
	double *dbFeaMat;
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
	double *updateWeightMat; /* stores the velocity in the weight space*/
	double *updateBiasMat;/* stores the velocity in the bias space*/
	FELink feaElem; /* stores the  input activations coming into the layer as well the output activations going out from the layer */
	ERLink errElem;
	TRLink info;
}LayerElem;

/*structure for ANN*/
typedef struct _ANNdef{
	int layerNum;
	LELink *layerList;/* list of layers*/
	OutFuncKind target; /* the activation function of the final output layer */
	ObjFuncKind errorfunc; /* the error function of the ANN */
	double *labelMat ; /* the target labels : BatchSample by targetDim matrix*/
}ANNDef;


void initialiseErrElems(ADLink anndef);
double drand();
double random_normal();
void initialiseBias(double *biasVec,int dim, int srcDim);
void initialiseWeights(double *weightMat,int length,int srcDim);
void initialiseLayer(LELink layer,int i, LELink srcLayer);
void  initialiseANN();

double computeTanh(double x);
double computeSigmoid(double x);
void computeActOfLayer(LELink layer);
void computeLinearActivation(LELink layer);
void fwdPassOfANN();

void computeDrvAct(double *dyfeat , double *yfeat,int len);
void computeActivationDrv (LELink layer);
void sumColsOfMatrix(double *dyFeatMat,double *dbFeatMat,int dim,int batchsamples);
void subtractMatrix(double *dyfeat, double* labels, int dim);
void CalcOutLayerBackwardSignal(LELink layer,ObjFuncKind errorfunc );
void BackPropBatch(ADLink anndef);

void findMaxElement(double *matrix, int row, int col, int *vec);
void updatateAcc(int *labels, LELink layer);
void addMatrixOrVec(double *weightMat, double* dwFeatMat, int dim);
void scaleMatrixOrVec(double* weightMat, double learningrate,int dim);
void updateNeuralNetParams(ADLink anndef, double lrnrate, double momentum, double weightdecay);
void updateLearningRate(int currentEpochIdx, double *lrnRate);
void TrainDNN(ADLink anndef);
void freeMemoryfromANN();
