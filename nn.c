#include <assert.h>
#include <stdarg.h>
#include <stdlib.h>

#include "engine.h"
#include "memory.h"
#include "nn.h"
#include "random.h"

static Neuron* newNeuron(int nin, bool nonlin) {
	Neuron* neuron = (Neuron*)malloc(sizeof(Neuron));
	neuron->nin = nin;
	neuron->nonlin = nonlin;
	neuron->W.values = (Value**)malloc(nin * sizeof(Value*));
	neuron->W.count = nin;
	neuron->W.capacity = nin;
	
	for (int i=0; i < nin; i++) {
		Value* val = newValue(randomUniform(-1.0, 1.0));
		neuron->W.values[i] = val;
	}
	neuron->b = newValue(0.0);

	return neuron;
}

static void freeNeuron(Neuron** neuron) {
	if (neuron == NULL || *neuron == NULL) {
		return;
	}
	Neuron* tempNeuron = *neuron;

	ValueArray* weights = &tempNeuron->W;
	for (int i=0; i < weights->count; i++) {
		freeValue(&weights->values[i]);
		assert(weights->values[i] == NULL);
	}
	free(weights->values);
	freeValue(&tempNeuron->b);
	free(tempNeuron);
	*neuron = NULL; 
}

static Value* forwardNeuron(Neuron* neuron, ValueArray* x) {
	Value* val = neuron->b; 
	for (int i=0; i < neuron->nin; i++) {
		val = vAdd(
			val,
			vMul(neuron->W.values[i], x->values[i])
		);
	}
	
	if (neuron->nonlin) {
		val = vRelu(val);
	}

	return val;
}

static ValueArray* paramsNeuron(Neuron* neuron) {
	ValueArray* array = (ValueArray*)malloc(sizeof(ValueArray));
	array->values = (Value**)malloc((neuron->nin + 1) * sizeof(Value*));
	array->count = neuron->nin + 1;
	array->capacity = neuron->nin + 1;

	for (int i=0; i < neuron->nin; i++) {
		array->values[i] = neuron->W.values[i];
	}
	array->values[neuron->nin] = neuron->b;
	return array;
}

static void initNeuronArray(NeuronArray* array) {
	array->values = NULL;
	array->capacity = 0;
	array->count = 0;
}

static void writeNeuronArray(NeuronArray* array, Neuron* neuron) {
	if (array->capacity < array->count + 1) {
		int oldCapacity = array->capacity;
		array->capacity = GROW_CAPACITY(oldCapacity);
		array->values = GROW_ARRAY(Neuron*, array->values, oldCapacity, array->capacity);
	}
	array->values[array->count] = neuron;
	array->count++;
}

static void freeNeuronArray(NeuronArray** array) {
	if (array == NULL || *array == NULL) {
		return;
	}
	
	NeuronArray* tempArray = *array;

	for (int i=0; i < tempArray->count; i++) {
		freeNeuron(&tempArray->values[i]);
		// assert(&tempArray->values[i] == NULL);
	}
	free(tempArray->values);
	initNeuronArray(tempArray);
	assert(tempArray->values == NULL);
	*array = NULL;
}

static Layer* newLayer(int nin, int nout, bool nonlin) {
	Layer* layer = (Layer*)malloc(sizeof(Layer));
	layer->nin = nin;
	layer->nout = nout;
	initNeuronArray(&layer->neurons);

	for (int i=0; i < layer->nout; i++) {
		writeNeuronArray(
			&layer->neurons,
			newNeuron(layer->nin, nonlin)
		);
	}
	return layer;
}

static void freeLayer(Layer** layer) { 
	if (layer == NULL || *layer == NULL) {
		return;
	}

	Layer* tempLayer = *layer;

	NeuronArray* neurons = &tempLayer->neurons;
	freeNeuronArray(&neurons);
	assert(neurons == NULL);
	free(tempLayer);
	*layer = NULL;
}

static ValueArray* forwardLayer(Layer* layer, ValueArray* x) {
	ValueArray* array = (ValueArray*)malloc(sizeof(ValueArray));
	array->values = (Value**)malloc(layer->nout * sizeof(Value*));
	array->count = layer->nout;
	array->capacity = layer->nout;

	
	for (int i=0; i < layer->nout; i++) {
		array->values[i] = forwardNeuron(layer->neurons.values[i], x);
	}

	return array;
}

static ValueArray* paramsLayer(Layer* layer) {
	ValueArray* array = (ValueArray*)malloc(sizeof(ValueArray));
	initValueArray(array);
	
	for (int i=0; i < layer->neurons.count; i++) {
		ValueArray* neuronParams = paramsNeuron(layer->neurons.values[i]);
		for (int j=0; j < neuronParams->count; j++) {
			writeValueArray(array, neuronParams->values[j])	;
		}
		freeValueArray(&neuronParams);
		assert(neuronParams == NULL);
	}

	return array;
}

static void initLayerArray(LayerArray* array) {
	array->values = NULL;
	array->capacity = 0;
	array->count = 0;
}

static void writeLayerArray(LayerArray* array, Layer* layer) {
	if (array->capacity < array->count + 1) {
		int oldCapacity = array->capacity;
		array->capacity = GROW_CAPACITY(oldCapacity);
		array->values = GROW_ARRAY(Layer*, array->values, oldCapacity, array->capacity);
	}
	array->values[array->count] = layer;
	array->count++;
}

static void freeLayerArray(LayerArray** array) {
	if (array == NULL || *array == NULL) {
		return;
	}

	LayerArray* tempArray = *array;

	for (int i=0; i < tempArray->count; i++) {
		freeLayer(&tempArray->values[i]);
		assert(tempArray->values[i] == NULL);
	}
	free(tempArray->values);
	initLayerArray(tempArray);
	assert(tempArray->values == NULL);

	*array = NULL;
}

MLP* newMLP(int count,...) {
	va_list args;
	va_start(args, count);
	
	// Copy arguments into argsArray
	int *argsArray = (int*)malloc(count * sizeof(int));
	if (argsArray == NULL) {
		fprintf(stderr, "Memoruy allocation failed\n");
		va_end(args);
		return NULL;
	}
	
	for (int i=0; i < count; i++) {
		argsArray[i] = va_arg(args, int);
	}

	va_end(args);

	// Initialize MLP
	MLP* mlp = (MLP*)malloc(sizeof(MLP));
	initLayerArray(&mlp->layers);
	
	for (int i=1; i < count; i++) {
		bool notLastLayer = i != count -1;
		writeLayerArray(&mlp->layers, newLayer(argsArray[i-1], argsArray[i], notLastLayer));
	}
	free(argsArray);
	
	return mlp;
}

Value* forwardMLP(MLP* mlp, ValueArray* x) {
	ValueArray* temp;
	for (int i=0; i < mlp->layers.count; i++) {
		temp = forwardLayer(mlp->layers.values[i], x);
		if (i > 0) freeValueArray(&x); // Do not free inputs
		x = temp;
	}
	
	Value* out = x->values[0];
	freeValueArray(&x);

	return out;
}

ValueArray* paramsMLP(MLP* mlp) {
	ValueArray* array = (ValueArray*)malloc(sizeof(ValueArray));
	initValueArray(array);
	
	for (int i=0; i < mlp->layers.count; i++) {
		ValueArray* layerParams = paramsLayer(mlp->layers.values[i]);
		for (int j=0; j < layerParams->count; j++) {
			writeValueArray(array, layerParams->values[j])	;
		}
		freeValueArray(&layerParams);
		assert(layerParams == NULL);
	}

	return array;
}

void freeMLP(MLP** mlp) {
	if (mlp == NULL || *mlp == NULL) {
		return;
	}

	MLP* tempMlp = *mlp;
	LayerArray* tempLayers = &tempMlp->layers;	
	freeLayerArray(&tempLayers);
	assert(tempLayers == NULL);
	free(tempMlp);
	*mlp = NULL;
}


void zeroGrad(ValueArray* params) {
	for (int i=0; i < params->count; i++) {
		params->values[i]->grad = 0;
	}
}

