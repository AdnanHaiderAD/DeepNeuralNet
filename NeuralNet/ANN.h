



enum ObjFuncKind = {XENT, SSE};
enum OutFuncKind = {REG, CLAS};
enum LayerRole = {HIDDEN,INPUT,OUTPUT};
enum ActFunKind = {SIGMOID,IDENTITY,TANH};

typedef struct _LayerElem *LELink;
typedef struct _ANNdef *ADLink;
typedef struct _FeatElem *FELink;
typedef struct _ErrorElem *ERLink;


typedef struct _ErrorElem{
	double dxfeatMat[];
	double dyFeatMat[];
}

typedef struct _FeatElem{
	double xfeatMat[];
	double yfeatMat[];
}FeaElem

/*structure for individual layers*/
typedef struct _LayerElem{
	int  layerId; /* each layer has a unique layer id */
	int  dim ; /* number of units in the hidden layer */
	LayerRole type; /* the type of layer : input,hidden or output */
	ActFunKind actfuncKind; /*each layer is allowed to have its own non-linear activation function */
	LayerElem* src; /* pointer to the input layer */
	int srcDim; /* the number of units in the input layer */
	double weights[];/* the weight matrix of the layer*/
	double bias[]; /* the bias vector */
	FELink feaElem; /* stores the  input activations coming into the layer as well the output activations going out from the layer */
	ERLink errElem;
}LayerElem;

/*structure for ANN*/
typedef struct _ANNdef{
	int layerNum;
	LayerInfo layerList;/* list of layers*/
	OutFuncKind outfunc; /* the activation function of the final output layer */
	ObjFuncKind errorfunc; /* the error function of the ANN */
}ANNDef;


void  computeActivationFunction(double *linearActivation,ActFunKind actfunc);
void  computeLayerActivation(*ANNDef anndef, int i);

void  intialiseWeightsNBias(*ANNDef anndef);
LELink initialiseLayer(int i, ActFunKind activefunc);
void  initialiseANN();
void  parseCMDArgs();



