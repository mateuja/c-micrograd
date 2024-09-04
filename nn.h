#ifndef micrograd_nn_h
#define micrograd_nn_h

#include <stdbool.h>
#include "engine.h"

typedef struct {
	int capacity;
	int count;
	float* values;
} FloatArray;

typedef struct {
	int nin;
	ValueArray W;
	Value* b;
	bool nonlin;
} Neuron;

typedef struct {
	int capacity;
	int count;
	Neuron** values;
} NeuronArray;

typedef struct {
	int nin;
	int nout;
	NeuronArray neurons;
} Layer;

typedef struct {
	int capacity;	
	int count;
	Layer** values;
} LayerArray;

typedef struct {
	LayerArray layers;
} MLP;

MLP* newMLP(int count, ...);
void freeMLP(MLP** mlp);
Value* forwardMLP(MLP* mlp, ValueArray* x);
ValueArray* paramsMLP(MLP* mlp);
void zeroGrad(ValueArray* params);

#endif

