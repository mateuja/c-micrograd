#include <assert.h>
#include <stdarg.h>
#include <stdlib.h>

#include "engine.h"
#include "nn.h"
#include "random.h"

static Neuron* newNeuron(size_t nin, bool nonlin) {
	Neuron* neuron = (Neuron*)malloc(sizeof(Neuron));
	neuron->nin = nin;
	neuron->nonlin = nonlin;
	neuron->W.items = (Value**)malloc(nin * sizeof(Value*));
	neuron->W.count = nin;
	neuron->W.capacity = nin;
	
	for (size_t i=0; i < nin; i++) {
		Value* val = newValue(randomUniform(-1.0, 1.0));
		neuron->W.items[i] = val;
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
	for (size_t i=0; i < weights->count; i++) {
		freeValue(&weights->items[i]);
		assert(weights->items[i] == NULL);
	}
	free(weights->items);
	freeValue(&tempNeuron->b);
	free(tempNeuron);
	*neuron = NULL; 
}

static Value* forwardNeuron(Neuron* neuron, ValueArray* x) {
	Value* val = neuron->b; 
	for (size_t i=0; i < neuron->nin; i++) {
		val = vAdd(
			val,
			vMul(neuron->W.items[i], x->items[i])
		);
	}
	
	if (neuron->nonlin) {
		val = vRelu(val);
	}

	return val;
}

static ValueArray* paramsNeuron(Neuron* neuron) {
	ValueArray* array = (ValueArray*)malloc(sizeof(ValueArray));
	array->items = (Value**)malloc((neuron->nin + 1) * sizeof(Value*));
	array->count = neuron->nin + 1;
	array->capacity = neuron->nin + 1;

	for (size_t i=0; i < neuron->nin; i++) {
		array->items[i] = neuron->W.items[i];
	}
	array->items[neuron->nin] = neuron->b;
	return array;
}

static void initNeuronArray(NeuronArray* array) {
	array->items = NULL;
	array->capacity = 0;
	array->count = 0;
}

static void freeNeuronArray(NeuronArray** array) {
	if (array == NULL || *array == NULL) {
		return;
	}
	
	NeuronArray* tempArray = *array;

	for (size_t i=0; i < tempArray->count; i++) {
		freeNeuron(&tempArray->items[i]);
		// assert(&tempArray->items[i] == NULL);
	}
	free(tempArray->items);
	initNeuronArray(tempArray);
	assert(tempArray->items == NULL);
	*array = NULL;
}

static Layer* newLayer(int nin, int nout, bool nonlin) {
	Layer* layer = (Layer*)malloc(sizeof(Layer));
	layer->nin = nin;
	layer->nout = nout;
	initNeuronArray(&layer->neurons);

	for (size_t i=0; i < layer->nout; i++) {
		APPEND_ARRAY(&(layer->neurons), newNeuron(layer->nin, nonlin));
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
	array->items = (Value**)malloc(layer->nout * sizeof(Value*));
	array->count = layer->nout;
	array->capacity = layer->nout;

	
	for (size_t i=0; i < layer->nout; i++) {
		array->items[i] = forwardNeuron(layer->neurons.items[i], x);
	}

	return array;
}

static ValueArray* paramsLayer(Layer* layer) {
	ValueArray* array = (ValueArray*)malloc(sizeof(ValueArray));
	initValueArray(array);
	
	for (size_t i=0; i < layer->neurons.count; i++) {
		ValueArray* neuronParams = paramsNeuron(layer->neurons.items[i]);
		for (size_t j=0; j < neuronParams->count; j++) {
			APPEND_ARRAY(array, neuronParams->items[j]);
		}
		freeValueArray(&neuronParams);
		assert(neuronParams == NULL);
	}

	return array;
}

static void initLayerArray(LayerArray* array) {
	array->items = NULL;
	array->capacity = 0;
	array->count = 0;
}

static void freeLayerArray(LayerArray** array) {
	if (array == NULL || *array == NULL) {
		return;
	}

	LayerArray* tempArray = *array;

	for (size_t i=0; i < tempArray->count; i++) {
		freeLayer(&tempArray->items[i]);
		assert(tempArray->items[i] == NULL);
	}
	free(tempArray->items);
	initLayerArray(tempArray);
	assert(tempArray->items == NULL);

	*array = NULL;
}

MLP* newMLP(size_t count,...) {
	va_list args;
	va_start(args, count);
	
	// Copy arguments into argsArray
	int *argsArray = (int*)malloc(count * sizeof(int));
	if (argsArray == NULL) {
		fprintf(stderr, "Memoruy allocation failed\n");
		va_end(args);
		return NULL;
	}
	
	for (size_t i=0; i < count; i++) {
		argsArray[i] = va_arg(args, int);
	}

	va_end(args);

	// Initialize MLP
	MLP* mlp = (MLP*)malloc(sizeof(MLP));
	initLayerArray(&mlp->layers);
	
	for (size_t i=1; i < count; i++) {
		bool notLastLayer = i != count -1;
		APPEND_ARRAY(&(mlp->layers), newLayer(argsArray[i-1], argsArray[i], notLastLayer));
	}
	free(argsArray);
	
	return mlp;
}

Value* forwardMLP(MLP* mlp, ValueArray* x) {
	ValueArray* temp;
	for (size_t i=0; i < mlp->layers.count; i++) {
		temp = forwardLayer(mlp->layers.items[i], x);
		if (i > 0) freeValueArray(&x); // Do not free inputs
		x = temp;
	}
	
	Value* out = x->items[0];
	freeValueArray(&x);

	return out;
}

ValueArray* paramsMLP(MLP* mlp) {
	ValueArray* array = (ValueArray*)malloc(sizeof(ValueArray));
	initValueArray(array);
	
	for (size_t i=0; i < mlp->layers.count; i++) {
		ValueArray* layerParams = paramsLayer(mlp->layers.items[i]);
		for (size_t j=0; j < layerParams->count; j++) {
			APPEND_ARRAY(array, layerParams->items[j]);
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
	for (size_t i=0; i < params->count; i++) {
		params->items[i]->grad = 0;
	}
}

