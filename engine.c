#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "engine.h"

int ID_COUNTER = 1;

void initValueArray(ValueArray* array) {
	array->items = NULL;
	array->capacity = 0;
	array->count = 0;
}

void freeValueArray(ValueArray** array) {
	if (array == NULL || *array == NULL) {
		return; // Do nothing if the array is already null
	}

	ValueArray* tempArray = *array;
	
	free(tempArray->items);
	tempArray->items = NULL;
	free(tempArray);
	tempArray = NULL;
	*array = NULL;
}

Value* newValue(float data) {
	Value* val = (Value*)malloc(sizeof(Value));
	if (val == NULL) {
		fprintf(stderr, "Memory allocation failed\n");
		exit(EXIT_FAILURE);
	}
	val->data = data;
	val->grad = 0;
	val->id = ID_COUNTER;
	val->backward = BW_NULL;
	ID_COUNTER++;
	initValueArray(&val->prev);

	return val;
}

void freeValue(Value** val) {
	if (val == NULL || *val == NULL) {
		return;
	}
	Value* tempValue = *val;	
	
	free(tempValue->prev.items);
	tempValue->prev.items = NULL;
	free(tempValue);
	*val = NULL;
}

void printValue(Value value) {
	printf("Value(id=%d, data=%.4f, grad=%.4f)\n", value.id, value.data, value.grad);
}

static void backwardAdd(Value* out, Value* fst, Value* snd)  {
	fst->grad += out->grad;
	snd->grad += out->grad;
}

Value* vAdd(Value* self, Value* other) {
	Value* out = newValue(self->data + other->data);
	out->backward = BW_ADD;
	APPEND_ARRAY(&out->prev, self);
	APPEND_ARRAY(&out->prev, other);
	return out;
}

Value* vAddFloat(Value* self, float other) {
	return vAdd(self, newValue(other));
}

static void backwardMul(Value* out, Value* fst, Value* snd) {
	fst->grad += snd->data * out->grad;
	snd->grad += fst->data * out->grad;
}

Value* vMul(Value* self, Value* other) {
	Value* out = newValue(self->data * other->data);
	out->backward = BW_MUL;
	APPEND_ARRAY(&out->prev, self);
	APPEND_ARRAY(&out->prev, other);
	return out;
}

Value* vMulFloat(Value* self, float other) {
	return vMul(self, newValue(other));
}

static void backwardPow(Value* out, Value* fst, Value* snd) {
	fst->grad += snd->data * pow(fst->data, snd->data - 1) * out->grad;
}

Value* vPow(Value* self, Value* other) {
	Value* out = newValue(pow(self->data, other->data));
	out->backward = BW_POW;
	APPEND_ARRAY(&out->prev, self);
	APPEND_ARRAY(&out->prev, other);
	return out;
}

Value* vPowFloat(Value* self, float other) {
	return vPow(self, newValue(other));
}

Value* vFloatPow(float self, Value* other) {
	return vPow(newValue(self), other);
}

static void backwardRelu(Value* out, Value* fst) {
	fst->grad += (out->data > 0 ? out->grad : 0);
}

Value* vRelu(Value* self) {
	Value* out = newValue(self->data < 0 ? 0 : self->data);
	out->backward = BW_RELU;
	APPEND_ARRAY(&out->prev, self);
	return out;
}

Value* vNeg(Value* self) {
	return vMulFloat(self, -1.0);
}

Value* vSub(Value* self, Value* other) {
	return vAdd(self, vNeg(other));
}

Value* vSubFloat(Value* self, float other) {
	return vAddFloat(self, -other);
}

Value* vDiv(Value* self, Value* other) {
	return vMul(self, vPowFloat(other, -1.0));
}

Value* vDivFloat(Value* self, float other) {
	return vDiv(self, newValue(other));
}

static void buildTopo(Value* val, ValueArray* topo, bool* visited) {
	if (!visited[val->id - 1]) {
		visited[val->id - 1] = true;
		for (size_t i=0; i < val->prev.count; i++) {
			buildTopo(val->prev.items[i], topo, visited);
		}
		APPEND_ARRAY(topo, val);
	}
}

void freeDAG(Value* last, int fromId) {
	bool* visited = (bool*)calloc((ID_COUNTER), sizeof(bool));
	ValueArray topo;
	initValueArray(&topo);

	buildTopo(last, &topo, visited);

	for (size_t i=0; i < topo.count; i++) {
		if (topo.items[i] != NULL && topo.items[i]->id >= fromId) {
			freeValue(&topo.items[i]);
			assert(topo.items[i] == NULL);
		}
	}
	free(topo.items);
	free(visited);
}

void backward(Value* val) {
	bool* visited = (bool*)calloc((ID_COUNTER), sizeof(bool));
	
	ValueArray topo;
	initValueArray(&topo);
	buildTopo(val, &topo, visited);

	val->grad = 1;
	
	Value* currentVal;
	size_t i = topo.count - 1;
	while (true) {
		currentVal = topo.items[i];
		
		switch (currentVal->backward) {
			case BW_NULL:
				break;
			case BW_ADD:
				backwardAdd(
					currentVal, currentVal->prev.items[0], currentVal->prev.items[1]
				);
				break;
			case BW_MUL:
				backwardMul(
					currentVal, currentVal->prev.items[0], currentVal->prev.items[1]
				);
				break;
			case BW_POW:
				backwardPow(
					currentVal, currentVal->prev.items[0], currentVal->prev.items[1]
				);
				break;
			case BW_RELU:
				backwardRelu(
					currentVal, currentVal->prev.items[0]
				);
				break;
		}
		if (i == 0) break;
		i--;

	}
	free(topo.items);
	free(visited);
}

