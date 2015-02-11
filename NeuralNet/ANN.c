#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "ANN.h"



static ActFunKind *actfunLists;
static int numHiddenLayers;
static OutFuncKind outputfunc;
static ADLINk anndef;










void  initialiseANN(){
	int i;
	anndef = malloc(sizeof(ANNDef));
	assert(anndef!=NULL);

	anndef->outfunc = outputfunc;
	anndef->layerNum = numHiddenLayers;
	for(i = 0; i<anndef->layerNum; i++){
		anndef->layerList[i] = 
	}

}
