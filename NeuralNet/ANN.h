typedef enum {XENT, SSE} ObjFuncKind;
typedef enum {REGRESSION, CLASSIFICATION} OutFuncKind;
typedef enum {HIDDEN,INPUT,OUTPUT} LayerRole;
typedef enum {SIGMOID,IDENTITY,TANH} ActFunKind;

typedef struct _LayerElem *LELink;
typedef struct _ANNdef *ADLink;
typedef struct _FeatElem *FELink;
typedef struct _ErrorElem *ERLink;


typedef struct _ErrorElem{
	double *dxfeatMat;
	double *dyFeatMat;
}ErrElem;

typedef struct _FeatElem{
	double *xfeatMat;
	double *yfeatMat;
}FeaElem;

/*structure for individual layers*/
typedef struct _LayerElem{
	int  id; /* each layer has a unique layer id */
	int  dim ; /* number of units in the hidden layer */
	LayerRole type; /* the type of layer : input,hidden or output */
	ActFunKind actfuncKind; /*each layer is allowed to have its own non-linear activation function */
	LELink src; /* pointer to the input layer */
	int srcDim; /* the number of units in the input layer */
	double *weights;/* the weight matrix of the layer*/
	double *bias; /* the bias vector */
	FELink feaElem; /* stores the  input activations coming into the layer as well the output activations going out from the layer */
	ERLink errElem;
}LayerElem;

/*structure for ANN*/
typedef struct _ANNdef{
	int layerNum;
	LELink *layerList;/* list of layers*/
	OutFuncKind outfunc; /* the activation function of the final output layer */
	ObjFuncKind errorfunc; /* the error function of the ANN */
}ANNDef;

void initialiseFeaElem(FELink feaElem, FELink srcElem);
double drand();
double random_normal();
void initialiseBias(double *biasVec,int dim, int srcDim);
void initialiseWeights(double *weightMat,int length,int srcDim);
void initialiseLayer(LELink layer,int i, LELink srcLayer);
void initialiseInputLayer(LELink inputlayer);
void  initialiseANN();

double computeTanh(double x);
double computeSigmoid(double x);
void computeActOfLayer(double* yfeatMat, int dim,  ActFunKind actfunc);
void computeLinearActivation(LELink layer);
void fwdPassOfANN();


void freeMemory();
