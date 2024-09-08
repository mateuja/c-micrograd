#ifndef micrograd_nn_h
#define micrograd_nn_h

#include <stdbool.h>
#include "engine.h"
#include "dynarray.h"

typedef struct {
	size_t nin;
	ValueArray W;
	Value* b;
	bool nonlin;
} Neuron;

DEFINE_DYNAMIC_ARRAY(Neuron);

typedef struct {
	size_t nin;
	size_t nout;
	NeuronArray neurons;
} Layer;

DEFINE_DYNAMIC_ARRAY(Layer);

typedef struct {
	LayerArray layers;
} MLP;

MLP* newMLP(size_t count, ...);
void freeMLP(MLP** mlp);
Value* forwardMLP(MLP* mlp, ValueArray* x);
ValueArray* paramsMLP(MLP* mlp);
void zeroGrad(ValueArray* params);

#endif

