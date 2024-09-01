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
	initValueArray(&neuron->W);
	
	for (int i=0; i < nin; i++) {
		Value* val = newValue(randomUniform(-1, 1));
		writeValueArray(&neuron->W, val);
	}
	neuron->b = newValue(0);

	return neuron;
}

static void freeNeuron(Neuron* neuron) {
	freeValueArray(&neuron->W);
	freeValue(neuron->b);
	free(neuron);
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
	for (int i=0; i < neuron->nin; i++) {
		writeValueArray(array, neuron->W.values[i]);
	}
	writeValueArray(array, neuron->b);
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

static void freeNeuronArray(NeuronArray* array) {
	for (int i=0; i < array->count; i++) {
		freeNeuron(array->values[i]);
	}
	free(array->values);
	initNeuronArray(array);
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

static void freeLayer(Layer* layer) {
	freeNeuronArray(&layer->neurons);
	free(layer);
}

static ValueArray* forwardLayer(Layer* layer, ValueArray* x) {
	ValueArray* array = (ValueArray*)malloc(sizeof(ValueArray));
	initValueArray(array);
	
	for (int i=0; i < layer->nout; i++) {
		writeValueArray(array, forwardNeuron(layer->neurons.values[i], x));
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
		freeValueArray(neuronParams);
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

static void freeLayerArray(LayerArray* array) {
	for (int i=0; i < array->count; i++) {
		freeLayer(array->values[i]);
	}
	free(array->values);
	initLayerArray(array);
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
		bool isLastLayer = i == count -1;
		writeLayerArray(&mlp->layers, newLayer(argsArray[i-1], argsArray[i], isLastLayer));
	}

	return mlp;
}

void freeMLP(MLP* mlp) {
	freeLayerArray(&mlp->layers);
	free(mlp);
}

Value* forwardMLP(MLP* mlp, ValueArray* x) {
	for (int i=0; i < mlp->layers.count; i++) {
		x = forwardLayer(mlp->layers.values[i], x);
	}
	return x->values[0];
}

ValueArray* paramsMLP(MLP* mlp) {
	ValueArray* array = (ValueArray*)malloc(sizeof(ValueArray));
	initValueArray(array);
	
	for (int i=0; i < mlp->layers.count; i++) {
		ValueArray* layerParams = paramsLayer(mlp->layers.values[i]);
		for (int j=0; j < layerParams->count; j++) {
			writeValueArray(array, layerParams->values[j])	;
		}
		freeValueArray(layerParams);
	}

	return array;
}

